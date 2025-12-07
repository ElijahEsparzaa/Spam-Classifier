import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import nltk
from nltk.corpus import stopwords

from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\rober\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"



# CONFIG
DATASET_PATH = os.path.join("data", "emails.csv")  # Make sure this file exists
TEST_SIZE = 0.2
RANDOM_STATE = 42



# LOAD DATASET
def load_dataset(path):
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found at: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    if "label" not in df.columns or "text" not in df.columns:
        print("[ERROR] CSV must have 'label' and 'text' columns.")
        sys.exit(1)

    df = df.dropna(subset=["text"])
    df = df.dropna(subset=["label"])

    # Map "spam" -> 1, "ham" -> 0 (or leave numeric if user provides it)
    label_map = {"spam": 1, "ham": 0}
    df["label"] = df["label"].map(label_map).fillna(df["label"]).astype(int)

    return df



# PIPELINE: TF-IDF + Naive Bayes
def build_pipeline():
    english_stopwords = stopwords.words("english")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=english_stopwords,
            max_df=0.95,
            min_df=2,
            ngram_range=(1,2)
        )),
        ("clf", MultinomialNB())
    ])

    return pipeline



# TRAIN + METRICS
def train_and_evaluate(df):
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()

    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    print("[INFO] Evaluating on test set...")
    y_pred = pipeline.predict(X_test)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    print("[INFO] Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print("CV accuracy scores:", cv_scores)
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}\n")

    return pipeline



# CLASSIFY TEXT
def classify_text(pipeline, text):
    if not text or text.strip() == "":
        return "invalid", 0.0

    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    spam_probability = float(proba[1])

    if pred == 1:
        return "SPAM", spam_probability
    else:
        return "HAM", spam_probability



# OCR IMAGE → TEXT
def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return ""

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return ""



# INTERFACE MENU
def interactive_menu(pipeline):
    print("\n==============================")
    print("  Spam Email Classifier (NB)")
    print("==============================")
    print("Safe interface: paste text or use screenshot OCR.\n")

    while True:
        print("Choose an option:")
        print("  1) Classify pasted email text")
        print("  2) Classify screenshot image (OCR)")
        print("  3) Exit")
        choice = input("Enter 1 / 2 / 3: ").strip()

        if choice == "1":
            print("\nPaste your email text. End with a blank line:")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)

            email_text = "\n".join(lines)
            label, prob = classify_text(pipeline, email_text)

            if label == "invalid":
                print("[WARN] No text entered.\n")
            else:
                print(f"\nPrediction: {label}")
                print(f"Spam probability: {prob:.4f}\n")

        elif choice == "2":
            image_path = input("Enter screenshot path (e.g., spam.png): ").strip()
            extracted = extract_text_from_image(image_path)

            if extracted.strip() == "":
                print("[WARN] No text extracted from image.\n")
                continue

            print("\nExtracted text (first 300 chars):")
            print(extracted[:300])
            print("\nClassifying...")

            label, prob = classify_text(pipeline, extracted)
            print(f"\nPrediction: {label}")
            print(f"Spam probability: {prob:.4f}\n")

        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("[WARN] Invalid option.\n")



# MAIN
def main():
    print("[INFO] Loading dataset...")
    df = load_dataset(DATASET_PATH)

    print(f"[INFO] Loaded {len(df)} samples\n")
    print(df["label"].value_counts())

    model = train_and_evaluate(df)

    interactive_menu(model)


if __name__ == "__main__":
    # make sure stopwords exist
    try:
        stopwords.words("english")
    except:
        nltk.download("stopwords")

    main()