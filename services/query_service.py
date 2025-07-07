from flask import Flask, request, jsonify
from flask_cors import CORS
from difflib import get_close_matches
from nltk.corpus import wordnet
from collections import Counter
import os

app = Flask(__name__)
CORS(app)

# 🧠 كلاس تحسين الاستعلام
class QueryRefiner:
    def __init__(self, processed_terms):
        self.term_frequencies = Counter(processed_terms)
        self.processed_terms = set(processed_terms)

    def suggest_correction(self, query):
        words = query.split()
        corrected = []
        for word in words:
            matches = get_close_matches(word, self.processed_terms, n=1, cutoff=0.8)
            if matches:
                corrected.append(matches[0])
            else:
                corrected.append(word)
        return corrected

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().lower()
                if name in self.processed_terms:
                    synonyms.add(name)
        return list(synonyms)

    def expand_query(self, words):
        expanded = set(words)
        for word in words:
            matches = get_close_matches(word, self.processed_terms, n=2, cutoff=0.7)
            expanded.update(matches)
            expanded.update(self.get_synonyms(word))
        return list(expanded)


# 🗂️ تحميل المفردات من ملف
def load_terms(dataset_name):
    dataset_key = dataset_name.replace("/", "_")
    vocab_file = os.path.join("..","vocabularies", f"{dataset_key}_terms.txt")
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    with open(vocab_file, "r", encoding="utf-8") as f:
        terms = [line.strip().lower() for line in f if line.strip()]
    return terms

@app.route("/enhance_query", methods=["POST"])
def enhance_query():
    data = request.get_json()
    query = data.get("query", "")
    dataset = data.get("dataset", "")
    if not query or not dataset:
        return jsonify({"error": "Both 'query' and 'dataset' are required"}), 400

    try:
        terms = load_terms(dataset)
        refiner = QueryRefiner(terms)

        print(f"🟨 Original query: {query}")
        corrected_words = refiner.suggest_correction(query)
        print(f"🟩 Corrected: {' '.join(corrected_words)}")

        expanded_words = refiner.expand_query(corrected_words)

        # حذف التكرارات والمحافظة على الكلمات الأصلية أو المصححة أولاً
        final_words = list(dict.fromkeys(corrected_words + expanded_words))

        # ترتيب وصياغة الاستعلام النهائي
        enhanced_query = " ".join(final_words)

        return jsonify({
            "original_query": query,
            "enhanced_query": enhanced_query
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(port=5007, debug=True)
