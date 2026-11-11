"""
Fake News Detector - a simple machine learning project
=======================================================

What it does
  1. Reads news articles that are already labelled FAKE or REAL
  2. Turns each article into numbers with TF-IDF (how important each word is)
  3. Trains a Logistic Regression model to tell FAKE from REAL
  4. Shows how accurate the model is on articles it has never seen
  5. Lets you type or paste any article and predicts if it is FAKE or REAL

How to run
  pip install pandas scikit-learn joblib streamlit
  streamlit run app.py                            the screen (recommended)
  python fake_news_detector.py                    console version: train, test, try articles
  python fake_news_detector.py --predict "text"   predict one article with the saved model

Data (put the files in the same folder as this script)
  Option A - Kaggle "Fake and Real News Dataset": Fake.csv and True.csv
  Option B - one file news.csv with the columns  text  and  label  (label = FAKE or REAL)
  No file found? A small built-in demo dataset is used so the script still runs.
"""

import os
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "fake_news_model.joblib")


# ----------------------------------------------------------------------------------------------
# 1. Load the data
# ----------------------------------------------------------------------------------------------
def demo_data():
    """A tiny example dataset, only so the script runs without downloading anything.
    The accuracy on it means nothing - use a real dataset for the project."""
    real = [
        "The central bank kept interest rates unchanged on Wednesday, citing steady inflation.",
        "The city council approved a new budget that increases funding for public schools.",
        "Scientists published a study showing a link between regular exercise and better sleep.",
        "The government announced road repairs on the main highway will finish next month.",
        "The company reported a five percent rise in quarterly profit compared with last year.",
        "Health officials recommend seasonal flu vaccines for older adults and young children.",
        "The national team won the match two goals to one after a late penalty.",
        "Heavy rain caused flooding in several towns and emergency services helped residents.",
        "The university opened a new research centre focused on renewable energy.",
        "Parliament passed a law requiring clearer food labels on packaged products.",
        "The space agency confirmed the satellite launch was successful on Tuesday morning.",
        "Local hospitals reported fewer emergency visits during the holiday weekend.",
        "The election commission published the final vote count for the regional election.",
        "Researchers found that a new battery design can charge electric cars faster.",
        "The museum will extend its opening hours during the summer festival.",
    ]
    fake = [
        "SHOCKING: Doctors don't want you to know this one fruit cures every disease overnight!",
        "Aliens secretly control the government, leaked documents prove it, share before deleted!",
        "Drinking hot water with lemon makes you immune to all viruses, scientists are hiding it!",
        "Celebrity reveals the moon landing was filmed in a basement, the truth is finally out!",
        "BREAKING: Banks will delete all savings accounts tomorrow, withdraw your money now!",
        "This miracle pill burns 20 kilos of fat in one week without diet or exercise!",
        "Secret study proves phones read your thoughts, they do not want you to see this!",
        "You won't believe what this politician did, the video they tried to ban!",
        "Eating chocolate every hour makes you smarter than a genius, experts stunned!",
        "Government plans to ban all cars by next Monday, insiders confirm the shocking plan!",
        "Wake up people! The weather is controlled by a machine hidden in the mountains!",
        "Miracle water from a secret spring makes people live 200 years, media silent!",
        "URGENT: New law will make breathing taxed starting next week, spread the word!",
        "Scientists admit the earth is flat in leaked email, the cover-up is over!",
        "This one weird trick makes you a millionaire in three days, banks hate it!",
    ]
    return pd.DataFrame({"text": real + fake, "label": ["REAL"] * len(real) + ["FAKE"] * len(fake)})


def load_data():
    fake_csv, true_csv = os.path.join(HERE, "Fake.csv"), os.path.join(HERE, "True.csv")
    news_csv = os.path.join(HERE, "news.csv")

    if os.path.exists(fake_csv) and os.path.exists(true_csv):
        print("Using Fake.csv and True.csv")
        fake = pd.read_csv(fake_csv)
        real = pd.read_csv(true_csv)
        fake["label"], real["label"] = "FAKE", "REAL"
        df = pd.concat([fake, real], ignore_index=True)
    elif os.path.exists(news_csv):
        print("Using news.csv")
        df = pd.read_csv(news_csv)
    else:
        print("No data file found - using the small built-in demo data.")
        print("For the real project, download the Kaggle 'Fake and Real News Dataset'.\n")
        return demo_data()

    # Use title + text when both exist
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df["label"] = df["label"].replace({"0": "REAL", "1": "FAKE", "TRUE": "REAL", "FALSE": "FAKE"})
    return df


# ----------------------------------------------------------------------------------------------
# 2. Clean the text
# ----------------------------------------------------------------------------------------------
def clean_text(text):
    text = str(text)
    # The Kaggle True.csv articles start with "CITY (Reuters) - ". Leaving that in would let the
    # model "cheat" by spotting the word Reuters instead of learning from the content.
    text = re.sub(r"^.{0,60}?\(Reuters\)\s*-\s*", "", text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)    # remove links
    text = re.sub(r"[^a-z\s]", " ", text)                  # keep letters only
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------------------------------
# 3. Train and test the model
# ----------------------------------------------------------------------------------------------
def build_and_evaluate(df):
    """Train on 80% of the articles, test on the other 20%.
    Returns the model and a dictionary with the results (used by the screen and the console)."""
    df = df.copy()
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]

    # 80% to learn from, 20% kept aside to test the model on articles it has never seen
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = ["FAKE", "REAL"]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fake_words, real_words = top_words(model)
    results = {
        "articles": len(df),
        "counts": df["label"].value_counts(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions),
        "report_table": pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).T,
        "confusion": pd.DataFrame(cm, index=["Actual " + l for l in labels],
                                  columns=["Predicted " + l for l in labels]),
        "fake_words": fake_words,
        "real_words": real_words,
    }
    return model, results


def train():
    """Console version: train, print the results and save the model."""
    df = load_data()
    model, r = build_and_evaluate(df)

    print("Articles:", r["articles"])
    print(r["counts"].to_string(), "\n")
    print("Accuracy: {:.2%}\n".format(r["accuracy"]))
    print(r["report"])
    print("Confusion matrix (rows = actual, columns = predicted)")
    print(r["confusion"].to_string(), "\n")
    print("Words that point to FAKE:", ", ".join(r["fake_words"]))
    print("Words that point to REAL:", ", ".join(r["real_words"]), "\n")

    joblib.dump(model, MODEL_FILE)
    print("Model saved to", MODEL_FILE)
    return model


def top_words(model, n=10):
    """The words that push the model most towards FAKE and towards REAL."""
    words = model.named_steps["tfidf"].get_feature_names_out()
    clf = model.named_steps["classifier"]
    order = clf.coef_[0].argsort()             # low weight -> classes_[0], high weight -> classes_[1]
    low, high = list(words[order[:n]]), list(words[order[-n:][::-1]])
    return (low, high) if clf.classes_[0] == "FAKE" else (high, low)


# ----------------------------------------------------------------------------------------------
# 4. Predict new articles
# ----------------------------------------------------------------------------------------------
def predict(model, article):
    probs = model.predict_proba([clean_text(article)])[0]
    best = probs.argmax()
    return model.classes_[best], probs[best]


def interactive(model):
    print("Paste an article (or a headline) and press Enter. Type 'quit' to stop.")
    while True:
        try:
            article = input("\nArticle: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if article.lower() in ("quit", "exit", "q"):
            break
        if article:
            label, confidence = predict(model, article)
            print("Prediction: {}  (confidence {:.0%})".format(label, confidence))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--predict":
        if not os.path.exists(MODEL_FILE):
            sys.exit("No saved model yet - run 'python fake_news_detector.py' first.")
        label, confidence = predict(joblib.load(MODEL_FILE), " ".join(sys.argv[2:]))
        print("Prediction: {}  (confidence {:.0%})".format(label, confidence))
    else:
        interactive(train())
