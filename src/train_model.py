import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import extract_text_from_file
from feature_extraction import clean_text, build_vectorizer

def train_model():
    df = pd.read_csv("data/labels.csv")
    texts, labels = [], df['label']

    for fn in df['filename']:
        text = extract_text_from_file(f"data/resumes/{fn}")
        texts.append(clean_text(text))

    vect = build_vectorizer(texts)
    X = vect.transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump({'vect': vect, 'clf': clf}, "model.pkl")
    print("✅ Model saved as model.pkl")

if __name__ == "__main__":
    train_model()
