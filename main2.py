from flask import Flask, request, jsonify
import requests
import joblib
import numpy as np
import os
import time
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import torch
import re
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)

def custom_tokenizer(text):
    return text.split()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
TFIDF_PATH = lambda dataset: os.path.join(BASE_DIR, "Data Representation", "TF-IDF", *dataset.split("/"), "doc", "tfidf_data.joblib")
BERT_PATH = lambda dataset: os.path.join(BASE_DIR, "Data Representation", "Bert", *dataset.split("/"), "doc", "bert_embedding.joblib")

@app.route("/searchuser", methods=["POST"])
def search_user():
    t_start = time.time()
    data = request.get_json()
    dataset = data.get("dataset")
    method = data.get("method")
    query = data.get("query")
    vector_store = data.get("vector_store", False)
    topics = data.get("topics", False)
    enhance = data.get("enhance", False)

    if not all([dataset, method, query]):
        return jsonify({"error": "dataset, method, and query are required"}), 400

    if method == "bert" and vector_store and topics:
        return jsonify({"error": "Cannot use both 'topics=true' and 'vector_store=true' together."}), 400

    # 1️⃣ استدعاء inverted index لترشيح المستندات أولاً
    try:
        inv_response = requests.post(
            "http://localhost:5010/filter_docs_by_query",
            json={"query": query, "dataset": dataset}
        )
        if inv_response.status_code != 200:
            return jsonify({"error": "Failed to get docs from inverted index"}), 500
        matched_doc_ids = inv_response.json().get("doc_ids", [])
    except Exception as e:
        return jsonify({"error": f"Inverted index service error: {e}"}), 500

    if not matched_doc_ids:
        return jsonify({
            "query": query,
            "method": method,
            "results": [],
            "message": "No matching documents found by inverted index"
        })

    # ✨ Enhance query if enabled
    if enhance:
        try:
            enhance_response = requests.post("http://localhost:5007/enhance_query", json={
                "query": query,
                "dataset": dataset,
                "top_k": 5
            })
            if enhance_response.status_code == 200:
                query = enhance_response.json().get("enhanced_query", query)
        except Exception:
            pass

    tfidf_indices = []
    bert_indices = []

    if method in ["tfidf", "hybrid"]:
        tfidf_data = joblib.load(TFIDF_PATH(dataset))
        tfidf_doc_ids = tfidf_data["doc_ids"]
        idx_map_tfidf = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
        tfidf_indices = [idx_map_tfidf[doc_id] for doc_id in matched_doc_ids if doc_id in idx_map_tfidf]

    if method in ["bert", "hybrid"]:
        bert_data = joblib.load(BERT_PATH(dataset))
        bert_doc_ids = bert_data.get("doc_ids")
        if not bert_doc_ids:
            return jsonify({"error": "BERT doc_ids not found"}), 500
        idx_map_bert = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}
        bert_indices = [idx_map_bert[doc_id] for doc_id in matched_doc_ids if doc_id in idx_map_bert]

    if method in ["tfidf", "hybrid"] and not tfidf_indices:
        return jsonify({
            "query": query,
            "method": method,
            "results": [],
            "message": "No matching documents found in TF-IDF indices"
        })
    if method in ["bert", "hybrid"] and not bert_indices:
        return jsonify({
            "query": query,
            "method": method,
            "results": [],
            "message": "No matching documents found in BERT indices"
        })

    # 🧠 الحسابات
    if method == "bert":
        response = requests.post("http://localhost:5002/compute_bert", json={
            "query": query,
            "dataset": dataset,
            "doc_indices": bert_indices
        })

    elif method == "tfidf":
        response = requests.post("http://localhost:5003/compute_tfidf", json={
            "query": query,
            "dataset": dataset,
            "doc_indices": tfidf_indices
        })

    elif method == "hybrid":
        response = requests.post("http://localhost:5004/compute_hybrid", json={
            "query": query,
            "dataset": dataset,
            "doc_indices_tfidf": tfidf_indices,
            "doc_indices_bert": bert_indices,
            "alpha": 0.5
        })

    else:
        return jsonify({"error": "Invalid method"}), 400

    if response.status_code != 200:
        return response.text, response.status_code

    results = response.json()["results"]

    # 🔁 ترجمة النتائج إلى doc_ids
    try:
        top_doc_ids = [matched_doc_ids[r["index"]] for r in results]
    except IndexError:
        return jsonify({"error": "Mismatch between result indices and matched_doc_ids"}), 500

    r_docs = requests.post("http://localhost:5005/get_docs_by_ids", json={
        "dataset": dataset,
        "doc_ids": top_doc_ids
    })

    if r_docs.status_code != 200:
        return r_docs.text, r_docs.status_code

    docs = {doc["doc_id"]: doc["text"] for doc in r_docs.json()}

    output = {
        "query": query,
        "method": method,
        "vector_store": False,
        "topics": False,
        "enhanced": enhance,
        "results": [
            {
                "doc_id": doc_id,
                "document": docs.get(doc_id, ""),
                "score": round(next((r["score"] for r in results if r["index"] == i), 6))
            }
            for i, doc_id in enumerate(top_doc_ids)
        ]
    }

    print(f"✅ TOTAL TIME: {round(time.time() - t_start, 2)} seconds\n")
    return jsonify(output)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
