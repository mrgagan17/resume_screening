import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 ]', '', text).lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop]
    return ' '.join(words)

def build_vectorizer(corpus):
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    vect.fit(corpus)
    return vect
