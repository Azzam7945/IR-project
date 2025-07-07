🔍 Information Retrieval System

This is a modular and extensible Information Retrieval (IR) system that supports multiple retrieval strategies including TF-IDF, BERT-based embedding, Hybrid methods, BERTopic-based filtering, and FAISS vector store for scalable vector search.
📁 Datasets Used

    antique/train

    antique/test

    beir/quora/test

🚀 Features

    🔎 TF-IDF search using sklearn's TfidfVectorizer

    🤖 BERT-based search using sentence-transformers/all-MiniLM-L6-v2

    ⚖️ Hybrid retrieval (TF-IDF + BERT)

    🧠 Topic-based filtering using BERTopic

    ⚡ FAISS support for efficient vector search

    🛠️ Query enhancement via paraphrasing

📦 Folder Structure
Folder	Description
ServerMongo/	Jupyter Notebook to initialize and insert data into MongoDB
Data Pre-Processing/	Prepares and cleans raw data
Data Representation/	Generates document representations using TF-IDF, BERT, and Hybrid
Indexing/	Creates index mappings and saves them
Query Processing/	Applies the same processing to queries as done to documents
vector_store/	FAISS indexing and searching for vector-based search
Topics/	BERTopic topic modeling + topic embeddings
Query Matching & Ranking/	Notebooks for scoring using TF-IDF, BERT, Hybrid, and evaluation metrics
services/	Flask-based microservices for each retrieval strategy
main2.py	Main API endpoint (port 5000) for unified search
index.html	Simple UI to test the search interactively
🧪 How to Use
1. Install dependencies and run MongoDB

    Install required libraries from requirements.txt

    Install and open MongoDB Compass

    Run ServerMongo.ipynb to populate the database

2. Data Preprocessing

cd Data\ Pre-Processing
# Run the notebook to clean and store data

3. Data Representation

Run the notebooks in this order:

    TF-IDF

    Bert

    Hybrid

4. Indexing

cd Indexing
# Run indexing notebook

5. Query Preprocessing

Same as data preprocessing but for queries:

cd Query\ Processing
# Run notebook

6. FAISS Vector Store

cd vector_store
# Run notebook to build FAISS index

7. BERTopic Modeling

cd Topics
# Run notebook to generate topics and topic embeddings

8. Query Matching & Ranking

Run notebooks in this order:

    TfidfMatching

    BertMatching

    HybridMatching

    Evaluation

9. Run Backend Services

Open a terminal in services/ and run each Python file (e.g. compute_bert.py, compute_tfidf.py, ...)

python compute_bert.py
python compute_tfidf.py
...

10. Run Main API

python main2.py

11. Launch Frontend

Open index.html in any browser.
🧠 Using the Search Interface

    Enter your query.

    Select dataset and method (TF-IDF / BERT / Hybrid).

    (Optional) Enable:

        ✅ Vector Store → Uses FAISS (only with BERT)

        ✨ Enhance Query → Paraphrases the input query

        🧠 Topics → Filters documents using predicted topic (only with BERT)

    Click Search and wait for results.

    ⚠️ You can’t use Vector Store and Topics together — choose only one.

🧠 BERT Model Used

All BERT-based retrieval is powered by:

sentence-transformers/all-MiniLM-L6-v2

