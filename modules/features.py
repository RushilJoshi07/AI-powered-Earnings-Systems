import pandas as pd
import numpy as np
import yfinance as yf
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


def get_sentence_level_scores(text):
    if not isinstance(text, str) or len(text) < 20:
        return 0.0, 0.0

    # cleaned text has no punctuation so we cannot split on periods
    # instead we split into fixed chunks of 30 words each
    # 30 words is roughly one sentence in financial speech
    words = text.split()
    chunks = [' '.join(words[i:i+30]) for i in range(0, len(words), 30)]
    chunks = [c for c in chunks if len(c.strip()) > 20]

    if not chunks:
        return 0.0, 0.0

    positive_sentences = 0
    negative_sentences = 0

    for chunk in chunks[:200]:
        try:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.numpy()[0]

            positive_prob = float(probs[0])
            negative_prob = float(probs[1])
            neutral_prob = float(probs[2])

            if neutral_prob > 0.6:
                continue
            elif positive_prob > negative_prob:
                positive_sentences += 1
            else:
                negative_sentences += 1

        except Exception:
            continue

    total_opinionated = positive_sentences + negative_sentences

    if total_opinionated == 0:
        return 0.0, 0.0

    optimism_ratio = round(positive_sentences / total_opinionated, 4)
    caution_ratio = round(negative_sentences / total_opinionated, 4)

    return optimism_ratio, caution_ratio


def get_price_volatility(symbol, date_str):
    # calculates annualised volatility of the stock in 30 days before earnings
    # high pre-earnings volatility means market was uncertain going into the call
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        start = date - timedelta(days=35)
        end = date - timedelta(days=1)

        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False
        )

        if df.empty or len(df) < 5:
            return None

        closes = df["Close"].dropna()
        if hasattr(closes.iloc[0], 'iloc'):
            closes = closes.iloc[:, 0]

        returns = closes.pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252))
        return round(volatility, 4)

    except Exception:
        return None


def get_prev_quarter_movement(symbol, date_str, all_records):
    # looks up what happened to the stock after the previous quarter earnings
    # uses our own dataset so no extra api calls needed
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        company_records = all_records[
            (all_records["symbol"] == symbol) &
            (pd.to_datetime(all_records["date"]) < date)
        ].sort_values("date", ascending=False)

        if company_records.empty:
            return None

        return round(float(company_records["movement_pct"].iloc[0]), 4)

    except Exception:
        return 0.0


def calculate_transcript_length(text):
    # word count of cleaned transcript
    # longer calls often signal more detailed guidance
    if not isinstance(text, str):
        return 0
    return len(text.split())


def calculate_word_complexity(text):
    # average word length as proxy for technical language depth
    if not isinstance(text, str):
        return 0
    words = text.split()
    if not words:
        return 0
    return round(sum(len(w) for w in words) / len(words), 4)


def calculate_sentiment_confidence(positive, negative, neutral):
    # measures how confident finbert was in its overall sentiment assessment
    # high confidence means finbert had a clear read on the transcript tone
    # low confidence means the transcript had genuinely mixed signals
    scores = [positive, negative, neutral]
    return round(float(max(scores) - min(scores)), 4)


def build_features():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "cleaned", "training_data_sentiment.csv"))

    print(f"building features for {len(df)} records")
    print("sentence level finbert scoring will take 20-30 minutes")

    feature_rows = []

    for idx, row in df.iterrows():
        symbol = row["symbol"]
        date_str = row["date"]
        text = row["cleaned_content"]

        # pure finbert sentence level signals
        optimism_ratio, caution_ratio = get_sentence_level_scores(text)

        # price based features
        volatility = get_price_volatility(symbol, date_str)
        prev_movement = get_prev_quarter_movement(symbol, date_str, df)

        # text structure features
        transcript_length = calculate_transcript_length(text)
        word_complexity = calculate_word_complexity(text)
        sentiment_confidence = calculate_sentiment_confidence(
            row["positive_score"],
            row["negative_score"],
            row["neutral_score"]
        )

        feature_row = {
            "symbol": symbol,
            "date": date_str,
            "quarter": row["quarter"],
            "positive_score": row["positive_score"],
            "negative_score": row["negative_score"],
            "neutral_score": row["neutral_score"],
            "polarity": row["polarity"],
            "sentiment_confidence": sentiment_confidence,
            "optimism_ratio": optimism_ratio,
            "caution_ratio": caution_ratio,
            "transcript_length": transcript_length,
            "word_complexity": word_complexity,
            "price_volatility": volatility,
            "prev_quarter_movement": prev_movement,
            "label": row["label"]
        }

        feature_rows.append(feature_row)

        if (idx + 1) % 25 == 0:
            print(f"processed {idx + 1} records")

    feature_df = pd.DataFrame(feature_rows)

    feature_df["price_volatility"] = feature_df["price_volatility"].fillna(
        feature_df["price_volatility"].median()
    )

    os.makedirs("data/features", exist_ok=True)
    output_path = os.path.join(BASE_DIR, "data", "features", "features_table.csv")
    feature_df.to_csv(output_path, index=False)

    print(f"saved {len(feature_df)} records to data/features/feature_table.csv")
    print(f"features: {[c for c in feature_df.columns if c not in ['symbol', 'date', 'quarter', 'label']]}")
    print(f"label distribution: {feature_df['label'].value_counts().to_dict()}")

    return feature_df


if __name__ == "__main__":
    build_features()