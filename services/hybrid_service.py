from flask import Flask, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import os

app = Flask(__name__)

def custom_tokenizer(text):
    return text.split()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ تحميل ملفات TF-IDF مرة واحدة عند بدء السيرفر
TFIDF_PATHS = {
    "antique/train": os.path.join(BASE_DIR, "Data Representation", "TF-IDF", "antique", "train", "doc", "tfidf_data.joblib"),
    "beir/quora/test": os.path.join(BASE_DIR, "Data Representation", "TF-IDF", "beir", "quora", "test", "doc", "tfidf_data.joblib")
}
tfidf_vectorizers = {
    dataset: joblib.load(path) for dataset, path in TFIDF_PATHS.items()
}

@app.route("/compute_hybrid", methods=["POST"])
def compute_hybrid():
    data = request.get_json()
    query = data["query"]
    dataset = data["dataset"]
    indices_tfidf = data["doc_indices_tfidf"]
    indices_bert = data["doc_indices_bert"]
    alpha = data.get("alpha", 0.5)

    if dataset not in tfidf_vectorizers:
        return jsonify({"error": "Invalid dataset"}), 400

    # 🟢 تحميل TF-IDF
    tfidf_data = tfidf_vectorizers[dataset]
    vectorizer = tfidf_data["vectorizer"]
    tfidf_matrix = tfidf_data["tfidf_matrix"]
    doc_tfidf = tfidf_matrix[indices_tfidf]

    # 🟢 تحميل BERT
    bert_path = os.path.join(BASE_DIR, "Data Representation", "Bert", *dataset.split("/"), "doc", "bert_embedding.joblib")
    bert_data = joblib.load(bert_path)
    embeddings_matrix = np.vstack(bert_data["embeddings_matrix"])
    doc_bert = embeddings_matrix[indices_bert]

    # 🟢 استخراج تمثيلات الاستعلام
    tfidf_query = vectorizer.transform([query])
    bert_query = model.encode(query)

    # 🔢 حساب التشابه
    tfidf_sim = cosine_similarity(tfidf_query, doc_tfidf)[0]
    bert_sim = cosine_similarity([bert_query], doc_bert)[0]
    hybrid_sim = alpha * tfidf_sim + (1 - alpha) * bert_sim

    # 🔝 أعلى النتائج
    top_indices = np.argpartition(hybrid_sim, -10)[-10:]
    sorted_indices = top_indices[np.argsort(-hybrid_sim[top_indices])]
    results = [{"index": int(i), "score": float(hybrid_sim[i])} for i in sorted_indices]

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(port=5004)
