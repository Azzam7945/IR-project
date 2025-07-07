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
    topics = data.get("topics", False)  # ✅ قراءة المتغير الجديد
    enhance = data.get("enhance", False)

    if not all([dataset, method, query]):
        return jsonify({"error": "dataset, method, and query are required"}), 400

    # ❌ التحقق من التوافق بين vector_store و topics
    if method == "bert" and vector_store and topics:
        return jsonify({"error": "Cannot use both 'topics=true' and 'vector_store=true' together."}), 400

    print("\n🔵 Received request")
    print(f"→ Dataset: {dataset} | Method: {method} | Query: {query} | Vector Store: {vector_store} | Topics: {topics} | Enhance: {enhance}")

    # ✅ Enhance query if enabled
    if enhance:
        try:
            print("✨ Enhancing the query via /enhance_query API ...")
            enhance_response = requests.post("http://localhost:5007/enhance_query", json={
                "query": query,
                "dataset": dataset,
                "top_k": 5
            })
            if enhance_response.status_code == 200:
                query = enhance_response.json().get("enhanced_query", query)
                print(f"🔧 Enhanced query: {query}")
            else:
                print("⚠️ Enhancement failed, using original query")
        except Exception as e:
            print(f"❌ Enhancement error: {e}, using original query")

    # ✅ TOPIC SEARCH MODE
    if method == "bert" and topics and not vector_store:
        print("🧠 Using topic-based filtering via /topic_search ...")
        try:
            response = requests.post("http://localhost:5008/topic_search", json={
                "query": query,
                "dataset": dataset
            })
            if response.status_code != 200:
                return response.text, response.status_code

            result_json = response.json()
            top_doc_ids = [item["doc_id"] for item in result_json["results"]]

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
                "topics": True,
                "vector_store": False,
                "enhanced": enhance,
                "predicted_topic": result_json.get("predicted_topic"),
                "topic_similarity_score": result_json.get("topic_similarity_score"),
                "results": [
                    {
                        "doc_id": item["doc_id"],
                        "document": docs.get(item["doc_id"], ""),
                        "score": round(item["score"], 6)
                    }
                    for item in result_json["results"]
                ]
            }

            print(f"✅ TOTAL TIME (Topic Filtering): {round(time.time() - t_start, 2)} seconds\n")
            return jsonify(output)

        except Exception as e:
            return jsonify({"error": f"Error during topic_search: {e}"}), 500

    # ✅ FAISS VECTOR STORE MODE
    if vector_store and method == "bert":
        print("📦 Using Vector Store with FAISS")
        bert_data = joblib.load(BERT_PATH(dataset))

        model_name = bert_data["model_name"]
        doc_ids = bert_data["doc_ids"]

        nltk.download('punkt')
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()

        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = text.split()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            return " ".join(tokens)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        def get_bert_vector(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
                return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        query_clean = clean_text(query)
        query_vector = get_bert_vector(query_clean)

        dataset_key = dataset.replace("/", "_")

        response = requests.post("http://localhost:8005/vector_search", json={
            "query_vector": query_vector.tolist(),
            "dataset_name": dataset_key,
            "top_k": 10
        })

        if response.status_code != 200:
            return response.text, response.status_code

        result_json = response.json()
        if result_json["status"] != "success":
            return jsonify(result_json), 500

        top_doc_ids = [item["doc_id"] for item in result_json["results"]]

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
            "vector_store": True,
            "topics": False,
            "enhanced": enhance,
            "results": [
                {
                    "doc_id": item["doc_id"],
                    "document": docs.get(item["doc_id"], ""),
                    "score": round(item["score"], 6)
                }
                for item in result_json["results"]
            ]
        }

        print(f"✅ TOTAL TIME (Vector Store): {round(time.time() - t_start, 2)} seconds\n")
        return jsonify(output)

    # 🟢 NORMAL MODE
    print("🟢 Step 1: Calling dataset loader ...")
    t1 = time.time()
    r1 = requests.post("http://localhost:5001/load_dataset", json={"dataset": dataset})
    if r1.status_code != 200:
        print("🔴 Error from dataset loader")
        return r1.text, r1.status_code
    common_doc_ids = r1.json()["common_doc_ids"]
    print(f"✅ Loaded {len(common_doc_ids)} common doc IDs in {round(time.time() - t1, 2)}s")

    tfidf_indices = []
    bert_indices = []

    if method in ["tfidf", "hybrid"]:
        print("📦 Loading TF-IDF model ...")
        tfidf_data = joblib.load(TFIDF_PATH(dataset))
        tfidf_doc_ids = tfidf_data["doc_ids"]
        idx_map_tfidf = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
        tfidf_indices = [idx_map_tfidf[doc_id] for doc_id in common_doc_ids]
        print(f"✅ TF-IDF indices ready")

    if method in ["bert", "hybrid"]:
        print("📦 Loading BERT embeddings ...")
        bert_data = joblib.load(BERT_PATH(dataset))
        bert_doc_ids = bert_data.get("doc_ids")
        if not bert_doc_ids:
            raise Exception("❌ 'doc_ids' not found in bert_data")
        idx_map_bert = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}
        bert_indices = [idx_map_bert[doc_id] for doc_id in common_doc_ids]
        print(f"✅ BERT indices ready")

    print(f"🛠 Step 3: Calling method service: {method}")
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
    top_doc_ids = [common_doc_ids[item["index"]] for item in results]

    r_docs = requests.post("http://localhost:5005/get_docs_by_ids", json={
        "dataset": dataset,
        "doc_ids": top_doc_ids
    })
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
                "score": round([r for r in results if common_doc_ids[r["index"]] == doc_id][0]["score"], 6)
            }
            for doc_id in top_doc_ids
        ]
    }

    print(f"✅ TOTAL TIME: {round(time.time() - t_start, 2)} seconds\n")
    return jsonify(output)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
