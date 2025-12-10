\# Complaint Clustering Engine



\## Project Description

This project automatically groups customer complaints into meaningful clusters without using predefined labels. The goal is to help humans understand common complaint themes easily.



\## Input Mechanism

\- Complaints are given through a CSV file (complaints.csv).

\- Each row contains one customer complaint.



\## How It Works

1\. Complaint text is cleaned and simplified.

2\. Text is converted into numerical form using TF-IDF.

3\. Similar complaints are grouped using clustering.

4\. Keywords and sample complaints are shown for interpretation.



\## Output

\- Each cluster shows:

&nbsp; - Important keywords

&nbsp; - Sample complaints

&nbsp; - Human-readable theme (assigned manually)



\## Interpretation

Clusters are not perfect labels. Users should read sample complaints and keywords to understand the theme of each cluster.



\## Conclusion

This system helps quickly identify major customer problem areas without manually reading all complaints.

