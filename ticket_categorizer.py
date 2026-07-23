"""
ticket_categorizer.py

AI/ML Intern Assessment - Auto Email / Ticket Categorizer
Author: Aayush Saini

What this does:
1. Loads a small dummy ticket dataset (subject + body + category)
2. Cleans the text (lowercase, remove punctuation/numbers, remove stopwords)
3. Converts text to TF-IDF features
4. Trains a Multinomial Naive Bayes classifier
5. Evaluates it (accuracy, precision/recall, confusion matrix)
6. Predicts category for 5 new sample tickets
7. Bonus: confidence score, low-confidence -> human review, priority tag,
   and a small CLI demo at the end

Note: The dataset used here is a dummy dataset I created myself for this
assessment (data/build_dataset.py). No real/public dataset is used.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# a small stopword list is enough here, did not want to add a heavy nltk download step
STOPWORDS = set("""
a an the is are was were am be been being to of in on for and or but if
this that these those i you he she it we they my your his her its our
their me him them us not no do does did doing have has had having
please can could will would should also just about with as at by from
""".split())


def clean_text(text):
    """Basic text cleaning: lowercase, remove punctuation/numbers, drop stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


# priority keywords for the bonus urgent/normal tag
URGENT_WORDS = ["down", "urgent", "not working", "asap", "immediately", "crash", "failed", "broken", "error"]


def get_priority(text):
    text_lower = text.lower()
    for word in URGENT_WORDS:
        if word in text_lower:
            return "Urgent"
    return "Normal"


def main():
    # ---------- 1. Load data ----------
    df = pd.read_csv("data/tickets.csv")
    print("Loaded", len(df), "tickets")
    print(df["category"].value_counts())
    print()

    # combine subject + body into one text field, this gives the model more context
    df["text"] = df["subject"] + " " + df["body"]
    df["clean_text"] = df["text"].apply(clean_text)

    # ---------- 2. Train/test split ----------
    X = df["clean_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- 3. TF-IDF vectorization ----------
    # TF-IDF is used instead of plain bag-of-words because it gives less weight
    # to common words that appear in almost every ticket (like "please", "issue")
    # and more weight to words that actually help tell categories apart.
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ---------- 4. Train model ----------
    # Naive Bayes is a common baseline for text classification and works well
    # even with a small dataset like this one, so it was picked over logistic
    # regression for the first version.
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # ---------- 5. Evaluate ----------
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print("Test set accuracy:", round(acc, 3))
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # save confusion matrix as an image for the report
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Ticket Categorizer")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("\nSaved confusion_matrix.png")

    # ---------- 6. Predict on 5 new sample tickets ----------
    # These are new tickets I wrote myself, not part of the training data.
    new_tickets = [
        "I was charged twice for my monthly subscription, please refund one of the charges",
        "The app keeps crashing every time I try to upload a photo, please help",
        "Can you tell me my remaining leave balance for this quarter",
        "What is the difference between your basic and premium plans",
        "Server is down and none of my api requests are working, this is urgent",
    ]

    print("\n---- Predictions on new sample tickets ----")
    results = predict_tickets(new_tickets, model, vectorizer)
    for r in results:
        print(f"\nTicket: {r['ticket']}")
        print(f"Predicted category: {r['category']}  (confidence: {r['confidence']}%)")
        print(f"Status: {r['status']}")
        print(f"Priority: {r['priority']}")

    # save these predictions to a csv as well, useful to attach in the report
    pred_df = pd.DataFrame(results)
    pred_df.to_csv("sample_predictions.csv", index=False)
    print("\nSaved sample_predictions.csv")

    return model, vectorizer


def predict_tickets(ticket_texts, model, vectorizer, confidence_threshold=60):
    """
    Takes a list of raw ticket texts and returns predicted category,
    confidence score, priority tag and whether it needs human review.
    confidence_threshold is in percentage (0-100).
    """
    results = []
    for ticket in ticket_texts:
        cleaned = clean_text(ticket)
        vec = vectorizer.transform([cleaned])

        probs = model.predict_proba(vec)[0]
        classes = model.classes_
        best_idx = probs.argmax()

        category = classes[best_idx]
        confidence = round(probs[best_idx] * 100, 1)

        # bonus: if confidence is low, flag for human review instead of auto-assigning
        status = "Auto-assigned" if confidence >= confidence_threshold else "Needs human review"

        priority = get_priority(ticket)

        results.append({
            "ticket": ticket,
            "category": category,
            "confidence": confidence,
            "status": status,
            "priority": priority,
        })
    return results


def cli_demo(model, vectorizer):
    """Simple command line demo - bonus objective. Type a ticket, get a category back."""
    print("\n---- Live demo ----")
    print("Type a support ticket and press enter to see the predicted category.")
    print("Type 'exit' to stop.\n")

    while True:
        text = input("Enter ticket text: ")
        if text.strip().lower() == "exit":
            break
        if text.strip() == "":
            continue
        result = predict_tickets([text], model, vectorizer)[0]
        print(f"-> Category: {result['category']} | Confidence: {result['confidence']}% "
              f"| Status: {result['status']} | Priority: {result['priority']}\n")


if __name__ == "__main__":
    trained_model, tfidf_vectorizer = main()

    # Uncomment the line below to try the interactive CLI demo
    # cli_demo(trained_model, tfidf_vectorizer)
