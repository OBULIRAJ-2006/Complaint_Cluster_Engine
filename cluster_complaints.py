import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------- DOWNLOAD REQUIRED NLTK DATA --------------------
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- STEP 1: LOAD DATA --------------------
df = pd.read_csv("complaints.csv")
texts = df["complaint_text"].astype(str).tolist()

# -------------------- STEP 2: TEXT PREPROCESSING --------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

cleaned_texts = [clean_text(t) for t in texts]

# -------------------- STEP 3: TEXT TO NUMBERS (TF-IDF) --------------------
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(cleaned_texts)

# -------------------- STEP 4: AUTOMATIC CLUSTER SELECTION --------------------
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

# -------------------- STEP 5: FINAL CLUSTERING --------------------
model = KMeans(n_clusters=k, random_state=42)
labels = model.fit_predict(X)

df["cluster"] = labels

# -------------------- STEP 6: EXTRACT KEYWORDS & SAMPLES --------------------
terms = vectorizer.get_feature_names_out()

output_data = []

for cluster_id in range(k):
    cluster_rows = df[df["cluster"] == cluster_id]
    cluster_texts = cluster_rows["complaint_text"].tolist()
    cluster_size = len(cluster_texts)

    mask = (df["cluster"].values == cluster_id)
    X_cluster = X[mask]
    mean_tfidf = X_cluster.mean(axis=0).A1

    top_indices = mean_tfidf.argsort()[-5:][::-1]
    top_keywords = [terms[i] for i in top_indices]

    output_data.append({
        "cluster": cluster_id,
        "size": cluster_size,
        "keywords": top_keywords,
        "samples": cluster_texts[:3]
    })

# -------------------- STEP 7: WRITE OUTPUT FILE --------------------
with open("clusters_output.txt", "w", encoding="utf-8") as f:
    f.write("COMPLAINT CLUSTERING ENGINE OUTPUT\n")
    f.write("=================================\n\n")

    for item in output_data:
        f.write(f"Cluster {item['cluster']}\n")
        f.write(f"Number of complaints: {item['size']}\n")
        f.write(f"Top Keywords: {', '.join(item['keywords'])}\n")
        f.write("Sample Complaints:\n")
        for s in item["samples"]:
            f.write(f"- {s}\n")
        f.write("\n---------------------------------\n\n")

print("Clustering completed successfully.")
print(f"Optimal number of clusters selected: {k}")
print("Check 'clusters_output.txt' for results.")
