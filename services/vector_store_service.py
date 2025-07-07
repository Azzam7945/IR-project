from flask import Flask, request, jsonify
import os
import faiss
import numpy as np
import joblib

app = Flask(__name__)

# 🔧 مجلد حفظ الفهارس
BASE_PATH = r"C:\Users\Azzam\PycharmProjects\PythonProject\vector_store\bert"

@app.route("/vector_search", methods=["POST"])
def vector_search():
    data = request.get_json()
    query_vector = data.get("query_vector")
    dataset_name = data.get("dataset_name")  # مثال: antique_train أو beir_quora_test
    top_k = data.get("top_k", 10)

    if query_vector is None or not dataset_name:
        return jsonify({"status": "error", "message": "Missing query_vector or dataset_name"}), 400

    # 🔎 تحميل الفهرس
    faiss_path = os.path.join(BASE_PATH, f"{dataset_name}.faiss")
    doc_ids_path = os.path.join(BASE_PATH, f"{dataset_name}_doc_ids.joblib")

    if not os.path.exists(faiss_path) or not os.path.exists(doc_ids_path):
        return jsonify({"status": "error", "message": "Index or doc_ids not found"}), 404

    try:
        index = faiss.read_index(faiss_path)
        doc_ids = joblib.load(doc_ids_path)

        # ⚙️ تهيئة الكويري
        query_np = np.array([query_vector], dtype=np.float32)
        if query_np.shape[1] != index.d:
            return jsonify({
                "status": "error",
                "message": f"Query vector dimension {query_np.shape[1]} does not match index dimension {index.d}"
            }), 400

        # 🔍 البحث
        distances, indices = index.search(query_np, top_k)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(doc_ids):
                results.append({
                    "doc_id": doc_ids[i],
                    "score": float(1 / (1 + dist))  # تحويل المسافة إلى تشابه
                })

        return jsonify({
            "status": "success",
            "results": results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Search failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(port=8005, debug=True)
