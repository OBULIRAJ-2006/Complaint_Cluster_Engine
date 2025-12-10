import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Step 1: Load complaints
df = pd.read_csv("complaints.csv")
texts = df["complaint_text"].tolist()

# Step 2: Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

cleaned_texts = [clean_text(t) for t in texts]

# Step 3: Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

# Step 4: Clustering
k = 4
model = KMeans(n_clusters=k, random_state=42)
labels = model.fit_predict(X)

df["cluster"] = labels

# Step 5: Get keywords per cluster
terms = vectorizer.get_feature_names_out()

output = []

for i in range(k):
    cluster_data = df[df["cluster"] == i]["complaint_text"].tolist()
    mask = (df["cluster"].values == i)   # boolean mask as NumPy array
    X_cluster = X[mask]                  # select only rows in this cluster
    tfidf_mean = X_cluster.mean(axis=0).A1
    top_terms = [terms[i] for i in tfidf_mean.argsort()[-5:][::-1]]

    output.append({
        "cluster": i,
        "keywords": top_terms,
        "samples": cluster_data[:3]
    })

# Step 6: Write output
with open("clusters_output.txt", "w") as f:
    for item in output:
        f.write(f"Cluster {item['cluster']}\n")
        f.write(f"Keywords: {', '.join(item['keywords'])}\n")
        f.write("Sample Complaints:\n")
        for s in item["samples"]:
            f.write(f"- {s}\n")
        f.write("\n---------------------------\n\n")

print("Clustering completed. Check clusters_output.txt")