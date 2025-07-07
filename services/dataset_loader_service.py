from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# الرجوع خطوة من مجلد services
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def custom_tokenizer(text):
    return text.split()


@app.route("/load_dataset", methods=["POST"])
def load_dataset():
    data = request.get_json()
    dataset_path = data.get("dataset")

    if dataset_path == "antique/train":
        tfidf_path = os.path.join(BASE_DIR, "Data Representation", "TF-IDF", "antique", "train", "doc", "tfidf_data.joblib")
        bert_path = os.path.join(BASE_DIR, "Data Representation", "Bert", "antique", "train", "doc", "bert_embedding.joblib")
    elif dataset_path == "beir/quora/test":
        tfidf_path = os.path.join(BASE_DIR, "Data Representation", "TF-IDF", "beir", "quora", "test", "doc", "tfidf_data.joblib")
        bert_path = os.path.join(BASE_DIR, "Data Representation", "Bert", "beir", "quora", "test", "doc", "bert_embedding.joblib")
    else:
        return jsonify({"error": "Invalid dataset"}), 400

    # Load TF-IDF
    tfidf_data = joblib.load(tfidf_path)
    tfidf_doc_ids = tfidf_data["doc_ids"]

    # Load BERT
    bert_data = joblib.load(bert_path)
    bert_doc_ids = bert_data.get("doc_ids", tfidf_doc_ids)

    common_doc_ids = list(set(tfidf_doc_ids).intersection(set(bert_doc_ids)))
    common_doc_ids.sort()

    return jsonify({
        "message": "Dataset loaded successfully",
        "common_doc_ids": common_doc_ids
    })


if __name__ == "__main__":
    app.run(port=5001, debug=True)
