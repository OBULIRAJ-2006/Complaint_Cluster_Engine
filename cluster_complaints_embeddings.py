import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------- NLTK SETUP --------------------
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- LOAD DATA --------------------
df = pd.read_csv("complaints.csv")
texts = df["complaint_text"].astype(str).tolist()

# -------------------- TEXT CLEANING --------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

cleaned_texts = [clean_text(t) for t in texts]

# -------------------- SENTENCE EMBEDDINGS --------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
X = embed_model.encode(cleaned_texts)

# -------------------- AUTO SELECT CLUSTERS --------------------
best_k = 2
best_score = -1

for k_test in range(2, min(8, len(texts))):
    model_test = KMeans(n_clusters=k_test, random_state=42)
    labels_test = model_test.fit_predict(X)
    score = silhouette_score(X, labels_test)

    if score > best_score:
        best_score = score
        best_k = k_test

k = best_k

# -------------------- FINAL CLUSTERING --------------------
model = KMeans(n_clusters=k, random_state=42)
labels = model.fit_predict(X)
df["cluster"] = labels

# -------------------- OUTPUT --------------------
with open("clusters_output.txt", "w", encoding="utf-8") as f:
    f.write("ADVANCED COMPLAINT CLUSTERING (BERT EMBEDDINGS)\n\n")

    for c in range(k):
        cluster_data = df[df["cluster"] == c]["complaint_text"].tolist()
        f.write(f"Cluster {c}\n")
        f.write(f"Number of complaints: {len(cluster_data)}\n")
        f.write("Sample Complaints:\n")
        for s in cluster_data[:4]:
            f.write(f"- {s}\n")
        f.write("\n--------------------------\n\n")

print("Advanced clustering completed. Check clusters_output.txt")