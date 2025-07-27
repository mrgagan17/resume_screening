# resume_screening

AI Resume Screening System
This project is a tool that reads resumes automatically and ranks them using Machine Learning.
It helps to select the best candidates faster.


Features
- PDF/DOCX resume parsing
- Text cleaning, lemmatization, stopword removal
- TF‑IDF features + Logistic Regression (or sentence‑transformers embeddings)
- Model trained to predict "hired" vs "not_hired"


How It Works

-Extracts text from resumes

-Uses TF‑IDF to convert text to numbers

-Trains Logistic Regression model

-Predicts score for new resumes and ranks them

-Upload resumes + enter job description
-Model reads and scores resumes using trained ML model
-Shows ranked candidates from best to worst

