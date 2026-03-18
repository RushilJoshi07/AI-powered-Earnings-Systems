import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

# load finbert model and tokenizer once at module level
# we load it here so it is only loaded once when the file is imported
# loading a transformer model takes 10-15 seconds so we never want to
# load it inside a function that gets called repeatedly
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# finbert outputs 3 classes in this exact order
LABELS = ["positive", "negative", "neutral"]


def chunk_text(text, chunk_size=400, overlap=50):
    # finbert has a maximum input of 512 tokens which is roughly 400 words
    # we split the transcript into overlapping chunks so no content is missed
    # overlap=50 means each chunk shares 50 words with the next chunk
    # this prevents important sentences from being cut off at chunk boundaries
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def analyze_chunk(text):
    # tokenize the text and convert to pytorch tensors
    # padding=True adds zeros to make all inputs the same length
    # truncation=True cuts off text longer than 512 tokens
    # return_tensors="pt" returns pytorch tensors not numpy arrays
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # torch.no_grad() tells pytorch not to calculate gradients
    # we are only doing inference not training so we do not need gradients
    # this saves memory and makes inference faster
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits are the raw scores for each class
    # softmax converts them to probabilities that sum to 1
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.numpy()[0]

    return {
        "positive": float(probs[0]),
        "negative": float(probs[1]),
        "neutral": float(probs[2])
    }


def analyze_transcript(text):
    # split transcript into chunks
    chunks = chunk_text(text)

    if not chunks:
        return None

    all_scores = []

    for chunk in chunks:
        if len(chunk.strip()) < 20:
            continue
        scores = analyze_chunk(chunk)
        all_scores.append(scores)

    if not all_scores:
        return None

    # average the scores across all chunks
    # this gives us one sentiment score for the entire transcript
    avg_positive = np.mean([s["positive"] for s in all_scores])
    avg_negative = np.mean([s["negative"] for s in all_scores])
    avg_neutral = np.mean([s["neutral"] for s in all_scores])

    # overall sentiment is whichever score is highest
    overall = LABELS[np.argmax([avg_positive, avg_negative, avg_neutral])]

    # polarity is a single number from -1 to +1
    # -1 means extremely negative, +1 means extremely positive
    # we calculate it as positive minus negative
    polarity = float(avg_positive - avg_negative)

    return {
        "positive_score": round(float(avg_positive), 4),
        "negative_score": round(float(avg_negative), 4),
        "neutral_score": round(float(avg_neutral), 4),
        "overall_sentiment": overall,
        "polarity": round(polarity, 4)
    }


def run_sentiment_analysis():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "training_data.csv"))

    print(f"running finbert on {len(df)} transcripts")
    print("this will take 15-25 minutes, finbert is processing each transcript chunk by chunk")

    results = []

    for idx, row in df.iterrows():
        text = row["cleaned_content"]

        if not isinstance(text, str) or len(text) < 50:
            results.append({
                "positive_score": None,
                "negative_score": None,
                "neutral_score": None,
                "overall_sentiment": None,
                "polarity": None
            })
            continue

        sentiment = analyze_transcript(text)

        if sentiment:
            results.append(sentiment)
        else:
            results.append({
                "positive_score": None,
                "negative_score": None,
                "neutral_score": None,
                "overall_sentiment": None,
                "polarity": None
            })

        if (idx + 1) % 50 == 0:
            print(f"processed {idx + 1} transcripts")

    sentiment_df = pd.DataFrame(results)
    final_df = pd.concat([df, sentiment_df], axis=1)
    final_df = final_df.dropna(subset=["positive_score"])

    output_path = os.path.join(BASE_DIR, "data", "cleaned", "training_data_cleaned.csv")
    final_df.to_csv(output_path, index=False)

    print(f"saved {len(final_df)} records with sentiment scores")
    print(f"average positive score: {final_df['positive_score'].mean():.4f}")
    print(f"average negative score: {final_df['negative_score'].mean():.4f}")
    print(f"average polarity: {final_df['polarity'].mean():.4f}")

    return final_df


if __name__ == "__main__":
    run_sentiment_analysis()