import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------- NLTK SETUP --------------------
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- STREAMLIT UI --------------------
st.title("Complaint Clustering Engine")
st.write("Automatically discover themes in customer complaints without predefined labels.")

uploaded_file = st.file_uploader("Upload complaints CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "complaint_text" not in df.columns:
        st.error("CSV must contain a column named 'complaint_text'")
    else:
        texts = df["complaint_text"].astype(str).tolist()

        # -------------------- TEXT PREPROCESSING --------------------
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            words = text.split()
            words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
            return " ".join(words)

        cleaned_texts = [clean_text(t) for t in texts]

        # -------------------- SENTENCE EMBEDDINGS (BERT) --------------------
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        X_embed = embed_model.encode(cleaned_texts)

        # -------------------- KEYWORD EXTRACTION (TF-IDF) --------------------
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        X_tfidf = tfidf_vectorizer.fit_transform(cleaned_texts)
        terms = tfidf_vectorizer.get_feature_names_out()

        # -------------------- CLUSTER SELECTION --------------------
        k = st.slider("Select number of clusters", min_value=2, max_value=8, value=4)

        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_embed)
        df["cluster"] = labels

        st.subheader("Clustering Results")

        # -------------------- DISPLAY CLUSTERS --------------------
        for c in range(k):
            st.markdown(f"### Cluster {c}")

            cluster_data = df[df["cluster"] == c]["complaint_text"].tolist()
            st.write(f"Number of complaints: {len(cluster_data)}")

            # Keywords
            mask = (df["cluster"].values == c)
            X_cluster = X_tfidf[mask]

            if X_cluster.shape[0] > 0:
                mean_tfidf = X_cluster.mean(axis=0).A1
                top_indices = mean_tfidf.argsort()[-5:][::-1]
                top_keywords = [terms[i] for i in top_indices]
                st.write("*Top Keywords:*", ", ".join(top_keywords))

            # Sample complaints
            st.write("*Sample Complaints:*")
            for s in cluster_data[:5]:
                st.write("-", s)

            st.write("---")