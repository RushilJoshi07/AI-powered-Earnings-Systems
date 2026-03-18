import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# load sentence transformer model once at module level
# this model converts text into vectors for semantic search
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# paths to our processed data files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTIMENT_PATH = os.path.join(BASE_DIR, "data", "cleaned", "training_data_sentiment.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features", "feature_table.csv")

# predefined taxonomy of financial themes
# groq picks from this list so results are always consistent
THEME_TAXONOMY = [
    "Revenue Growth", "Gross Margin", "Operating Margin", "Net Income",
    "EPS Beat", "EPS Miss", "Guidance Raised", "Guidance Lowered",
    "Guidance Maintained", "AI Investment", "Cloud Computing", "Supply Chain",
    "Demand Weakness", "Demand Strength", "Interest Rates", "Inflation Impact",
    "Cost Cutting", "Headcount Reduction", "Hiring Plans", "M&A Activity",
    "Share Buyback", "Dividend Increase", "Debt Management", "Cash Flow",
    "International Expansion", "China Exposure", "Regulatory Risk",
    "Product Launch", "R&D Investment", "Competitive Pressure",
    "Consumer Spending", "Commercial Real Estate", "Credit Quality",
    "Loan Growth", "Net Interest Margin", "Energy Transition",
    "Healthcare Innovation", "Cybersecurity", "Inventory Management",
    "Pricing Power"
]


def lookup_company(ticker):
    # searches our processed data for the given ticker
    # returns the most recent transcript and all associated data
    try:
        df = pd.read_csv(SENTIMENT_PATH)
        company_data = df[df["symbol"] == ticker.upper()].copy()

        if company_data.empty:
            return None

        # sort by date and get most recent quarter
        company_data["date"] = pd.to_datetime(company_data["date"])
        company_data = company_data.sort_values("date", ascending=False)
        latest = company_data.iloc[0]

        # get stock movement data from features file
        features_df = pd.read_csv(FEATURES_PATH)
        feature_row = features_df[
            (features_df["symbol"] == ticker.upper()) &
            (features_df["date"] == latest["date"].strftime("%Y-%m-%d"))
        ]

        movement_pct = float(feature_row["prev_quarter_movement"].iloc[0]) if not feature_row.empty else None
        price_volatility = float(feature_row["price_volatility"].iloc[0]) if not feature_row.empty else None

        return {
            "symbol": ticker.upper(),
            "date": latest["date"].strftime("%Y-%m-%d"),
            "quarter": latest["quarter"],
            "content": latest["content"],
            "cleaned_content": latest["cleaned_content"],
            "positive_score": float(latest["positive_score"]),
            "negative_score": float(latest["negative_score"]),
            "neutral_score": float(latest["neutral_score"]),
            "polarity": float(latest["polarity"]),
            "overall_sentiment": latest["overall_sentiment"],
            "movement_pct": movement_pct,
            "price_volatility": price_volatility
        }

    except Exception as e:
        print(f"error looking up {ticker}: {e}")
        return None


def build_faiss_index(transcript_text):
    # splits transcript into 30 word chunks
    # converts each chunk to a vector
    # builds a searchable FAISS index
    words = transcript_text.split()
    chunks = []
    for i in range(0, len(words), 30):
        chunk = " ".join(words[i:i+30])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)

    if not chunks:
        return None, []

    # convert all chunks to vectors at once
    # this is faster than converting one by one
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    # build faiss index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


def search_transcript(question, index, chunks, top_k=4):
    # converts the question to a vector
    # finds the most similar chunks in the faiss index
    # returns the most relevant transcript passages
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]

    return relevant_chunks


def extract_themes(transcript_text):
    # sends transcript to groq
    # groq picks the most relevant themes from our predefined taxonomy
    try:
        # send first 3000 words to keep within groq token limits
        words = transcript_text.split()
        truncated = " ".join(words[:3000])

        themes_str = ", ".join(THEME_TAXONOMY)

        prompt = f"""You are a financial analyst. Read this earnings call transcript excerpt and identify the 6 to 8 most relevant themes discussed.

You must only choose from this exact list of themes:
{themes_str}

Transcript excerpt:
{truncated}

Return only a comma separated list of theme names from the list above. Nothing else. No explanation. No numbering."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )

        themes_raw = response.choices[0].message.content.strip()
        themes = [t.strip() for t in themes_raw.split(",") if t.strip() in THEME_TAXONOMY]

        return themes if themes else ["Revenue Growth", "Operating Margin"]

    except Exception as e:
        print(f"error extracting themes: {e}")
        return ["Revenue Growth", "Operating Margin"]


def generate_reasoning(company_data, relevant_chunks):
    try:
        chunks_text = "\n\n".join(relevant_chunks[:5])

        movement = company_data.get('movement_pct')
        movement_str = f"{movement}%" if movement is not None else "not available"

        prompt = f"""You are a senior financial analyst writing a brief earnings call summary for a non-financial audience.

You have the following data:
- Company: {company_data['symbol']}
- Quarter: {company_data['quarter']}
- Earnings date: {company_data['date']}
- Overall sentiment tone: {company_data['overall_sentiment']}
- Positive language score: {company_data['positive_score']:.1%}
- Negative language score: {company_data['negative_score']:.1%}
- Stock movement after earnings: {movement_str}

Relevant transcript excerpts:
{chunks_text}

Write a 4 to 5 sentence earnings summary that does the following:
1. Summarize what management talked about and how they sounded overall
2. Highlight the most important thing management said based on the excerpts
3. Briefly note whether the stock movement aligns with the tone of the call
4. Use simple plain English that anyone can understand — no jargon

Do not start with "Based on". Do not say "it is unclear". Be direct and informative."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=400
        )

        reasoning = response.choices[0].message.content.strip()
        reasoning = reasoning.replace("`", "$")
        return reasoning

    except Exception as e:
        print(f"error generating reasoning: {e}")
        return "Unable to generate analysis at this time."


def get_quarter_trend(ticker):
    try:
        df = pd.read_csv(SENTIMENT_PATH)
        company_data = df[df["symbol"] == ticker.upper()].copy()

        if len(company_data) < 3:
            return None

        company_data["date"] = pd.to_datetime(company_data["date"])
        company_data = company_data.sort_values("date", ascending=True)

        # load features for movement data
        features_df = pd.read_csv(FEATURES_PATH)
        company_features = features_df[features_df["symbol"] == ticker.upper()].copy()

        trend = []
        for _, row in company_data.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")

            # get movement for this quarter from features
            feat_row = company_features[company_features["date"] == date_str]
            movement = float(feat_row["prev_quarter_movement"].iloc[0]) if not feat_row.empty else None

            # calculate overall tone from polarity threshold
            polarity = float(row["polarity"])
            if polarity > 0.05:
                tone = "Positive"
            elif polarity < -0.05:
                tone = "Negative"
            else:
                tone = "Neutral"

            trend.append({
                "quarter": row["quarter"],
                "date": date_str,
                "polarity": polarity,
                "positive_score": float(row["positive_score"]),
                "negative_score": float(row["negative_score"]),
                "overall_sentiment": tone,
                "movement_pct": movement
            })

        return trend

    except Exception as e:
        print(f"error getting quarter trend for {ticker}: {e}")
        return None


def answer_question(question, index, chunks):
    try:
        if not question or len(question.strip()) < 5:
            return "Please ask a more specific question about the earnings call."

        relevant_chunks = search_transcript(question, index, chunks)

        if not relevant_chunks:
            return "Could not find relevant information in the transcript for your question."

        context = "\n\n".join(relevant_chunks)

        prompt = f"""You are a financial analyst assistant helping a non-financial person understand an earnings call.

Answer the question below using the transcript excerpts provided. Give a detailed, clear answer of at least 3 to 4 sentences. Explain any financial terms in simple language. If the transcript does not contain a direct answer, say what related information you did find and explain it clearly.

Transcript excerpts:
{context}

Question: {question}

Give a thorough, helpful answer:"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()
        answer = answer.replace("`", "$")
        return answer

    except Exception as e:
        print(f"error answering question: {e}")
        return "Unable to process your question at this time."


def analyze_company(ticker):
    # master function that ties everything together
    # called by the dashboard when a user searches a ticker
    # returns everything needed to populate the full dashboard
    print(f"analyzing {ticker}")

    company_data = lookup_company(ticker)
    if company_data is None:
        return None

    print("building faiss index")
    index, chunks = build_faiss_index(company_data["content"])

    print("extracting themes")
    themes = extract_themes(company_data["content"])

    print("generating reasoning")
    relevant_chunks = search_transcript(
        f"what did management say about performance guidance and outlook",
        index,
        chunks
    )
    reasoning = generate_reasoning(company_data, relevant_chunks)

    print("getting quarter trend")
    trend = get_quarter_trend(ticker)

    return {
        "company_data": company_data,
        "index": index,
        "chunks": chunks,
        "themes": themes,
        "reasoning": reasoning,
        "trend": trend
    }


if __name__ == "__main__":
    ticker = input("enter ticker to test: ").strip().upper()
    result = analyze_company(ticker)

    if result:
        print(f"symbol: {result['company_data']['symbol']}")
        print(f"quarter: {result['company_data']['quarter']}")
        print(f"overall sentiment: {result['company_data']['overall_sentiment']}")
        print(f"polarity: {result['company_data']['polarity']}")
        print(f"themes: {result['themes']}")
        print(f"reasoning: {result['reasoning']}")
        print(f"trend quarters available: {len(result['trend']) if result['trend'] else 0}")
    else:
        print("company not found in dataset")