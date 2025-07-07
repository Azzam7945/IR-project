from flask import Flask, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import os

app = Flask(__name__)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BERT_PATHS = {
    "beir/quora/test": os.path.join(BASE_DIR, "Data Representation", "Bert", "beir", "quora", "test", "doc", "bert_embedding.joblib"),
    "antique/train": os.path.join(BASE_DIR, "Data Representation", "Bert", "antique", "train", "doc", "bert_embedding.joblib")
}

@app.route("/compute_bert", methods=["POST"])
def compute_bert():
    data = request.get_json()
    query = data["query"]
    dataset = data["dataset"]
    indices = data["doc_indices"]

    print("🟢 Loading BERT data for:", dataset)
    bert_data = joblib.load(BERT_PATHS[dataset])
    bert_embeddings = np.vstack(bert_data["embeddings_matrix"])
    selected_embeddings = bert_embeddings[indices]

    print(f"📐 Selected {len(indices)} embeddings")

    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], selected_embeddings)[0]

    top_indices = np.argpartition(similarities, -10)[-10:]
    sorted_indices = top_indices[np.argsort(-similarities[top_indices])]
    results = [{"index": int(i), "score": float(similarities[i])} for i in sorted_indices]

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(port=5002)
