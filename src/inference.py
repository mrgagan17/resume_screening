import joblib
from utils import extract_text_from_file
from feature_extraction import clean_text

model = joblib.load("model.pkl")
vect, clf = model["vect"], model["clf"]

def score_resume(path):
    text = extract_text_from_file(path)
    text = clean_text(text)
    X = vect.transform([text])
    prob = clf.predict_proba(X)[0][clf.classes_ == "hired"][0]
    return float(prob)
