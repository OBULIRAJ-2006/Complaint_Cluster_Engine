import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------- NLTK SETUP --------------------
nltk.download("stopwords")
nltk.download("wordnet")

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Complaint Clustering Engine", layout="centered")
st.title("Complaint Clustering Engine")
st.write("Unsupervised Machine Learning for Complaint Analysis")

# -------------------- SESSION STATE --------------------
if "complaints" not in st.session_state:
    st.session_state.complaints = []

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = set()

if "cluster_names" not in st.session_state:
    st.session_state.cluster_names = {}

# -------------------- RESET --------------------
if st.button("Reset Complaints"):
    st.session_state.complaints = []
    st.session_state.loaded_files = set()
    st.session_state.cluster_names = {}
    st.rerun()

# -------------------- ADD REAL-TIME COMPLAINT --------------------
st.subheader("Add Complaint")

new_complaint = st.text_input("Enter complaint text")

if st.button("Add Complaint"):
    if new_complaint.strip() and new_complaint not in st.session_state.complaints:
        st.session_state.complaints.append(new_complaint)
        st.success("Complaint added")

# -------------------- MULTIPLE CSV UPLOAD (CORRECT) --------------------
uploaded_files = st.file_uploader(
    "Upload complaints CSV files (column: complaint_text)",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.loaded_files:
            df_upload = pd.read_csv(file)

            if "complaint_text" in df_upload.columns:
                for text in df_upload["complaint_text"].astype(str).tolist():
                    if text not in st.session_state.complaints:
                        st.session_state.complaints.append(text)

                st.session_state.loaded_files.add(file.name)
                st.success(f"{file.name} loaded")
            else:
                st.error(f"{file.name} missing 'complaint_text' column")

# -------------------- STOP IF NOT ENOUGH DATA --------------------
if len(st.session_state.complaints) < 2:
    st.info("Add at least 2 complaints to start clustering")
    st.stop()

# -------------------- TEXT PREPROCESSING --------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

cleaned_texts = [clean_text(t) for t in st.session_state.complaints]

# -------------------- BERT EMBEDDINGS --------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
X_embed = embed_model.encode(cleaned_texts)

# -------------------- CLUSTERING --------------------
k = st.slider(
    "Number of clusters",
    min_value=2,
    max_value=min(10, len(cleaned_texts)),
    value=3
)

kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_embed)

df = pd.DataFrame({
    "complaint_text": st.session_state.complaints,
    "cluster": labels
})

# -------------------- DELETE (SINGLE CONTROL) --------------------
st.subheader("Delete a Complaint")

cluster_to_delete = st.selectbox(
    "Select Cluster",
    sorted(df["cluster"].unique())
)

complaints_in_cluster = df[df["cluster"] == cluster_to_delete]["complaint_text"].tolist()

complaint_to_delete = st.selectbox(
    "Select Complaint",
    complaints_in_cluster
)

if st.button("Delete Selected Complaint"):
    st.session_state.complaints.remove(complaint_to_delete)
    st.success("Complaint deleted")
    st.rerun()

# -------------------- CLUSTER NAMING --------------------
st.subheader("Name Clusters (Optional)")

for c in sorted(df["cluster"].unique()):
    name = st.text_input(
        f"Name for Cluster {c}",
        value=st.session_state.cluster_names.get(c, ""),
        key=f"cluster_name_{c}"
    )
    if name.strip():
        st.session_state.cluster_names[c] = name

# -------------------- TF-IDF KEYWORDS --------------------
tfidf = TfidfVectorizer(max_features=800)
X_tfidf = tfidf.fit_transform(cleaned_texts)
terms = tfidf.get_feature_names_out()

# -------------------- SMALL CLEAN CHART --------------------
st.subheader("Complaint Distribution")

counts = df["cluster"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(counts.index.astype(str), counts.values)
ax.set_xlabel("Cluster")
ax.set_ylabel("Complaints")
ax.set_title("Complaints per Cluster", fontsize=10)
ax.tick_params(labelsize=8)
st.pyplot(fig)

# -------------------- CLUSTER DETAILS (OLD STYLE) --------------------
st.subheader("Cluster Details")

for c in range(k):
    title = st.session_state.cluster_names.get(c, f"Cluster {c}")
    st.markdown(f"### {title}")

    cluster_texts = df[df["cluster"] == c]["complaint_text"].tolist()
    st.write(f"**Number of complaints:** {len(cluster_texts)}")

    mask = (df["cluster"] == c).values
    X_cluster = X_tfidf[mask]

    if X_cluster.shape[0] > 0:
        mean_tfidf = X_cluster.mean(axis=0).A1
        keywords = [terms[i] for i in mean_tfidf.argsort()[-5:][::-1]]
        st.write("**Top Keywords:**", ", ".join(keywords))

    for text in cluster_texts:
        st.write("-", text)

    st.markdown("---")
