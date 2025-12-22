## Complaint Clustering Engine

## Project Overview

The Complaint Clustering Engine is a machine learning project that helps organize customer complaints into meaningful groups automatically. Instead of manually reading and sorting complaints, this system uses unsupervised machine learning to discover common problem areas from raw complaint text.

The main goal of this project is understanding and interpretation, not just accuracy. The clusters help identify what customers are actually struggling with.


---

## Problem Statement

Companies receive a large number of customer complaints every day through different platforms like apps, emails, and customer support systems. Reading and categorizing these complaints manually is slow and difficult, especially when the number of complaints grows.

Also, complaint categories are often not clearly defined in advance, making automation challenging.


---

## Solution Approach

This project provides an automated way to analyze complaints by:

Accepting raw complaint text
Finding hidden patterns using machine learning
Grouping similar complaints together
Helping users understand each group through keywords and examples

The system works without any predefined labels, making it flexible for different datasets.


---

## Key Features

Upload one or more CSV files containing complaints
Add complaints manually in real time
Automatically group complaints using machine learning
Visualize how complaints are distributed across clusters
Allow users to name clusters in simple words
Delete incorrect or irrelevant complaints easily
Display all complaints inside each cluster



---

## Input Format

The system accepts CSV files with a single column named:

complaint_text

Example:

Payment failed but money was deducted
Delivery arrived later than promised
App crashes after login

Complaints can also be added directly through the dashboard.


---

## How Machine Learning Is Used

Text Cleaning

Each complaint is cleaned by:

Converting text to lowercase
Removing special characters
Removing common stopwords
Converting words to their base form


This makes the text easier for the model to understand.


---

## Understanding Meaning with BERT

The project uses a pretrained BERT-based model to convert each complaint into a numerical form that represents its meaning. This allows the system to group complaints that mean the same thing even if the wording is different.

For example:

Amount debited but payment failed
Money deducted but transaction unsuccessful


These will likely be grouped together.


---

## Clustering with K-Means

Once the complaints are converted into vectors, K-Means clustering is applied. This algorithm groups similar complaints together without knowing the categories beforehand.

The user can control how many clusters they want based on the dataset size.


---

## Making Clusters Understandable

Since machine learning does not label clusters automatically:

Important keywords are extracted for each cluster
All complaints in a cluster are displayed
Users assign meaningful names to each cluster


This makes the results easy to interpret.


---

## Output

For each cluster, the system shows:

Cluster name (given by the user)
Number of complaints
Key terms related to the issue
All complaints belonging to that cluster
A bar chart showing complaint distribution



---

## Real-Time Usage

Complaints can be added while the system is running. When new complaints are added:

Clusters update automatically
Charts refresh instantly
Emerging issues can be detected early



---

## Managing Complaints

Users can remove a specific complaint by selecting:

1. The cluster

2. The complaint

3. Clicking delete

This helps keep the data clean and relevant.


---

## Tools and Technologies Used

Python
Streamlit for the web interface
SentenceTransformers (BERT) for text understanding
Scikit-learn for clustering
NLTK for text preprocessing
Matplotlib for visualization



---

## How to Run the Project

One-Time Setup
pip install streamlit pandas scikit-learn sentence-transformers nltk matplotlib
Run the Application
python -m streamlit run app.py


---

## Important Note

This project is designed to support human analysis. Clusters may change when new complaints are added or removed. Human interpretation plays a key role in understanding and naming clusters.


---

## Possible Future Improvements

Automatically suggest cluster names
Track complaint trends over time
Add alerts for sudden complaint spikes
Support complaints in multiple languages

---

## Conclusion

The Complaint Clustering Engine demonstrates how unsupervised machine learning can be used to turn unstructured customer complaints into meaningful insights. It focuses on clarity, usability, and real-world applicability rather than just technical accuracy.

The Complaint Clustering Engine demonstrates how unsupervised machine learning can be used to turn unstructured customer complaints into meaningful insights. It focuses on clarity, usability, and real-world applicability rather than just technical accuracy.
