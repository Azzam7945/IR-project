from flask import Flask, request, jsonify
from pymongo import MongoClient

app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017/")
db = client["ir_project"]

@app.route("/get_docs_by_ids", methods=["POST"])
def get_docs_by_ids():
    data = request.get_json()
    dataset = data["dataset"]
    ids = data["doc_ids"]

    if dataset == "antique/train":
        collection = db["documents_test"]
    elif dataset == "beir/quora/test":
        collection = db["documents_quora_test"]
    else:
        return jsonify({"error": "Invalid dataset"}), 400

    docs = collection.find({"doc_id": {"$in": ids}}, {"_id": 0, "doc_id": 1, "original_text": 1})
    result = [{"doc_id": d["doc_id"], "text": d["original_text"]} for d in docs]

    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5005)
