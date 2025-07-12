from flask import Flask, request, jsonify
import json
import os
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 📁 تحديد مسار مجلد الفهارس
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Indexing")

# 🧠 تحميل الفهارس المعكوسة عند بدء التشغيل
inverted_indexes = {}
AVAILABLE_DATASETS = ["beir/quora/test", "antique/train"]

for dataset in AVAILABLE_DATASETS:
    dataset_key = dataset.replace("/", "_")  # لتكوين اسم ملف الفهرس
    index_file = os.path.join(BASE_DIR, f"{dataset_key}_inverted_index.json")

    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            inverted_indexes[dataset] = json.load(f)
        print(f"✅ Loaded inverted index for: {dataset}")
    else:
        print(f"⚠️ Missing index file for dataset: {dataset}")


# 🧹 دالة لتقسيم الاستعلام إلى كلمات بسيطة
def tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


# 🟢 نقطة النهاية: استرجاع معرفات المستندات المطابقة للاستعلام
@app.route("/filter_docs_by_query", methods=["POST"])
def filter_docs_by_query():
    data = request.get_json()
    query = data.get("query")
    dataset = data.get("dataset")

    if not query or not dataset:
        return jsonify({"error": "Both 'query' and 'dataset' are required"}), 400

    if dataset not in inverted_indexes:
        return jsonify({"error": f"No index found for dataset: {dataset}"}), 400

    tokens = tokenize(query)
    index = inverted_indexes[dataset]

    matched_doc_ids = set()
    for token in tokens:
        if token in index:
            matched_doc_ids.update(index[token])

    return jsonify({"doc_ids": list(matched_doc_ids)})


# 🚀 تشغيل السيرفر
if __name__ == "__main__":
    app.run(port=5010, debug=True)
