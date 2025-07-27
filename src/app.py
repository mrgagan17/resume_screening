from flask import Flask, request, render_template
from inference import score_resume
from utils import save_uploaded_file
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    job_desc = request.form.get("job_description")
    files = request.files.getlist("resumes")

    results = []
    for f in files:
        path = save_uploaded_file(f, UPLOAD_FOLDER)
        score = score_resume(path)
        results.append({"filename": f.filename, "score": round(score, 2)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return render_template("result.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
