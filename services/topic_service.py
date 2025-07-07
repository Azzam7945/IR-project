from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🛠️ المسارات
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOPIC_MODEL_PATH = lambda dataset: os.path.join(PROJECT_ROOT, "TopicResults", f"{dataset.replace('/', '_')}_bertopic_results.joblib")
BERT_DATA_PATH = lambda dataset: os.path.join(PROJECT_ROOT, "Data Representation", "Bert", *dataset.split("/"), "doc", "bert_embedding.joblib")

# 🧠 تحميل نموذج التضمين
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/topic_search", methods=["POST"])
def topic_search():
    data = request.get_json()
    query = data.get("query")
    dataset = data.get("dataset")

    if not query or not dataset:
        return jsonify({"error": "Both 'query' and 'dataset' are required"}), 400

    print(f"\n🔍 Received query: '{query}' for dataset: '{dataset}'")

    # 1️⃣ تحميل البيانات
    try:
        topic_data = joblib.load(TOPIC_MODEL_PATH(dataset))
        bert_data = joblib.load(BERT_DATA_PATH(dataset))
        print("✅ Loaded topic model and BERT embeddings.")
    except Exception as e:
        return jsonify({"error": f"Error loading data: {e}"}), 500

    topics = topic_data["topics"]
    doc_ids = topic_data["doc_ids"]
    topic_embeddings = topic_data["topic_embeddings"]  # 🧠 تمثيل التوبيكات
    all_doc_embeddings = bert_data["embeddings_matrix"]
    all_doc_ids = bert_data["doc_ids"]

    # 2️⃣ تمثيل الاستعلام
    print("📌 Encoding query into BERT representation...")
    query_embedding = embedding_model.encode([query])[0]

    # 3️⃣ مطابقة الاستعلام مع كل التوبيكات
    print("🤝 Matching query with topic embeddings...")
    topic_ids = list(topic_embeddings.keys())
    topic_vectors = np.array([topic_embeddings[tid] for tid in topic_ids])
    topic_similarities = cosine_similarity([query_embedding], topic_vectors)[0]

    best_topic_index = np.argmax(topic_similarities)
    predicted_topic = topic_ids[best_topic_index]
    best_score = topic_similarities[best_topic_index]

    print(f"🏷️ Best matching topic: {predicted_topic} (cosine sim: {best_score:.4f})")

    # 4️⃣ استخراج الوثائق ضمن التوبيك المحدد
    print(f"🔎 Filtering documents for topic {predicted_topic}...")
    topic_doc_indices = [i for i, t in enumerate(topics) if t == predicted_topic]

    if not topic_doc_indices:
        print("⚠️ No documents found for this topic.")
        return jsonify({"results": [], "message": "No documents found for this topic"}), 200

    topic_doc_embeddings = [all_doc_embeddings[i] for i in topic_doc_indices]
    topic_doc_ids = [doc_ids[i] for i in topic_doc_indices]
    print(f"📄 Found {len(topic_doc_ids)} documents in topic {predicted_topic}.")

    # 5️⃣ حساب التشابه بين الاستعلام والوثائق داخل التوبيك
    print("📈 Calculating similarity between query and documents in the topic...")
    similarities = cosine_similarity([query_embedding], topic_doc_embeddings)[0]
    top_k = min(10, len(similarities))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i in top_indices:
        doc_id = topic_doc_ids[i]
        score = float(round(similarities[i], 6))
        results.append({"doc_id": doc_id, "score": score})
        print(f"   → doc_id: {doc_id}, score: {score}")

    # 6️⃣ إرجاع النتيجة
    return jsonify({
        "query": query,
        "predicted_topic": int(predicted_topic),
        "topic_similarity_score": float(round(best_score, 6)),
        "results": results
    })

if __name__ == "__main__":
    app.run(port=5008, debug=True)
