from flask import Flask, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

app = Flask(__name__)


def custom_tokenizer(text):
    return text.split()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# تحميل ملفات TF-IDF مسبقًا
TFIDF_PATHS = {
    "antique/train": os.path.join(BASE_DIR, "Data Representation", "TF-IDF", "antique", "train", "doc", "tfidf_data.joblib"),
    "beir/quora/test": os.path.join(BASE_DIR, "Data Representation", "TF-IDF", "beir", "quora", "test", "doc", "tfidf_data.joblib")
}

tfidf_vectorizers = {
    dataset: joblib.load(path) for dataset, path in TFIDF_PATHS.items()
}

@app.route("/compute_tfidf", methods=["POST"])
def compute_tfidf():
    data = request.get_json()
    query = data["query"]
    dataset = data["dataset"]
    indices = data["doc_indices"]

    if dataset not in tfidf_vectorizers:
        return jsonify({"error": "Invalid dataset"}), 400

    tfidf_data = tfidf_vectorizers[dataset]
    vectorizer = tfidf_data["vectorizer"]
    tfidf_matrix = tfidf_data["tfidf_matrix"]

    # استخراج فقط المستندات المطلوبة
    matrix = tfidf_matrix[indices]

    # استخراج تمثيل الاستعلام
    query_vec = vectorizer.transform([query])

    # حساب التشابه
    similarities = cosine_similarity(query_vec, matrix)[0]

    # ترتيب النتائج
    top_indices = np.argpartition(similarities, -10)[-10:]
    sorted_indices = top_indices[np.argsort(-similarities[top_indices])]
    results = [{"index": int(i), "score": float(similarities[i])} for i in sorted_indices]

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(port=5003)
