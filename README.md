# Complaint Clustering Engine

## Project Description
The Complaint Clustering Engine is an unsupervised machine learning system that
automatically groups customer complaints into meaningful clusters without using
predefined labels. The goal of the project is to help humans quickly understand
common complaint themes using cluster keywords and sample complaints.

---

## Problem Statement
When the number of customer complaints is large, manual analysis becomes
time-consuming and inefficient. There is a need for a system that can automatically
identify common complaint themes without requiring predefined categories.

---

## Input Mechanism
- Input is provided through a CSV file.
- The CSV file must contain a column named complaint_text.
- Each row represents a single customer complaint.

---

## Methodology

1. Customer complaint text is collected and loaded from a CSV file.
2. Text preprocessing is applied:
   - Conversion to lowercase
   - Removal of special characters
   - Stopword removal
   - Lemmatization
3. Cleaned text is converted into numerical form using sentence embeddings
   generated from a pre-trained BERT model.
4. K-Means clustering is applied to group similar complaints without predefined labels.
5. TF-IDF is used only for extracting important keywords from each cluster.
6. Sample complaints and keywords are displayed to support human interpretation.

---

## Machine Learning Used

- *Type:* Unsupervised Machine Learning
- *Text Representation:* BERT-based Sentence Embeddings (all-MiniLM-L6-v2)
- *Clustering Algorithm:* K-Means
- *Keyword Extraction:* TF-IDF

The machine learning model groups complaints based on semantic similarity rather
than exact word matching.

---

## Interactive Dashboard

A Streamlit-based dashboard is provided where users can:
- Upload a complaint dataset
- Select the number of clusters
- View cluster-wise complaint distribution
- Analyze top keywords and sample complaints

This makes the system interactive and user-friendly.

---

## Interpretation Note
The clustering results are not final labels. Users must interpret each cluster by
analyzing the keywords and sample complaints and assign human-readable theme names.
Interpretation is more important than perfect clustering.

---

## Output
For each cluster, the system displays:
- Cluster number
- Number of complaints
- Top keywords
- Sample complaints

---

## How to Run

1. Install required libraries:
   pip install pandas scikit-learn nltk sentence-transformers streamlit

2. Run the application:
   python -m streamlit run app.py

3. Upload a CSV file containing complaint data.

---

## Conclusion
This project demonstrates how unsupervised machine learning can be used to discover
hidden patterns in customer complaints. By combining semantic embeddings, clustering,
and human interpretation, the system provides meaningful insights without requiring
predefined labels.

---

## Future Enhancements
- Automatic theme naming
- Trend analysis over time
- Multilingual complaint support
- Advanced visual analytics
