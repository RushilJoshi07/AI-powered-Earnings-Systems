import re
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
import os

nlp = spacy.load("en_core_web_sm")
STOPWORDS = set(stopwords.words("english"))

# these are words that appear constantly in earnings calls but carry
# zero financial signal — operator instructions, pleasantries, legal boilerplate
# we remove them so FinBERT focuses only on meaningful financial language
EARNINGS_NOISE = {
    "operator", "thank", "thanks", "welcome", "call", "conference",
    "question", "answer", "please", "go", "ahead", "next", "caller",
    "line", "open", "hold", "moment", "recording", "replay", "forward",
    "looking", "statement", "statements", "safe", "harbor", "cautionary",
    "remarks", "prepared", "presentation", "slide", "slides", "page",
    "turning", "referring", "refer", "see", "noted", "note", "noted",
    "hello", "hi", "good", "morning", "afternoon", "evening", "everyone",
    "everybody", "today", "joining", "joined", "participate", "participation",
    "introducing", "introduce", "moderator", "host", "hosting"
}


def remove_speaker_labels(text):
    # earnings transcripts have lines like:
    # "Tim Cook -- Chief Executive Officer"
    # "John Smith - Analyst, Goldman Sachs"
    # these are just speaker introductions, not financial content
    # the pattern looks for a name followed by -- or - and a title
    text = re.sub(r'^[A-Z][a-zA-Z\s]+--[^\n]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[A-Z][a-zA-Z\s]+-[^\n]+$', '', text, flags=re.MULTILINE)
    return text


def remove_boilerplate(text):
    # only remove the section header labels themselves
    # not the content inside those sections
    text = re.sub(r'Prepared Remarks:', '', text)
    text = re.sub(r'Questions and Answers:', '', text)
    text = re.sub(r'Question-and-Answer Session', '', text)
    return text


def clean_text(text):
    # step 1 — remove speaker labels like "Tim Cook -- CEO"
    text = remove_speaker_labels(text)

    # step 2 — remove boilerplate sections
    text = remove_boilerplate(text)

    # step 3 — lowercase everything
    # "Revenue" and "revenue" should be treated as the same word
    text = text.lower()

    # step 4 — remove urls if any exist in the text
    text = re.sub(r'http\S+|www\S+', '', text)

    # step 5 — remove numbers and currency symbols
    # "$4.2 billion" becomes "" — the sentiment model cares about
    # the words around numbers not the numbers themselves
    text = re.sub(r'\$[\d,.]+', '', text)
    text = re.sub(r'[\d,.]+%', '', text)
    text = re.sub(r'\d+', '', text)

    # step 6 — remove punctuation and special characters
    # keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # step 7 — remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords_and_noise(text):
    words = text.split()

    # remove standard english stopwords (the, is, and, etc)
    # and our custom earnings call noise words
    filtered = [
        word for word in words
        if word not in STOPWORDS and word not in EARNINGS_NOISE and len(word) > 2
    ]

    return " ".join(filtered)


def lemmatize(text):
    # lemmatization converts words to their base form
    # "growing" -> "grow", "revenues" -> "revenue", "invested" -> "invest"
    # this means the model treats these as the same concept
    # we process in chunks of 50000 characters because spacy has a max length limit
    doc = nlp(text[:50000])
    lemmatized = " ".join([token.lemma_ for token in doc if not token.is_space])
    return lemmatized


def clean_transcript(text):
    # this is the master function that runs all cleaning steps in order
    # every transcript passes through this single function
    if not text or not isinstance(text, str):
        return ""

    text = clean_text(text)
    text = remove_stopwords_and_noise(text)
    text = lemmatize(text)

    return text


def clean_dataset():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "training_data.csv"))

    print(f"cleaning {len(df)} transcripts")

    df["cleaned_content"] = df["content"].apply(clean_transcript)

    # remove any rows where cleaning produced empty text
    df = df[df["cleaned_content"].str.len() > 100]

    output_path = os.path.join(BASE_DIR, "data", "cleaned", "training_data_cleaned.csv")
    df.to_csv(output_path, index=False)

    print(f"saved {len(df)} cleaned records")
    print(f"sample original length: {len(df['content'].iloc[0])} characters")
    print(f"sample cleaned length: {len(df['cleaned_content'].iloc[0])} characters")

    return df


if __name__ == "__main__":
    clean_dataset()