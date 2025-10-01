
# Legal Document Analysis and Classification Project

## Overview

This project focuses on the collection, analysis, and classification of legal documents related to decisions of the Supreme Court of Greece (Άρειος Πάγος) and Greek legal codes. It is divided into two main parts:

---

## Part A: Data Collection of Supreme Court Decisions

In the first part, we develop a web crawling tool to collect data from the official website of the Supreme Court of Greece ([areiospagos.gr](https://areiospagos.gr)) focusing on decisions issued in 2024. The scope includes both criminal and civil cases.

### Objectives:

* Extract relevant metadata for each decision, including:

  * Decision number and year
  * Department type (Civil or Criminal)
  * Department number (e.g., Β2)
  * Judges involved
  * Court introduction text
  * Legal reasoning ("ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ")
  * Relevant Penal Code (ΠΚ), Criminal Procedure Code (ΚΠΔ) articles for criminal cases
  * Relevant Civil Code (ΑΚ), Civil Procedure Code (ΚΠολΔ) articles for civil cases

* Organize the extracted data into a structured dataframe based on specifications derived from the Huggingface GreekLegalSum dataset for consistency.

### Tasks:

* Implement and run web scraping and crawling scripts.
* Analyze the collected data against expected datasets (e.g., GreekLegalSum).
* Visualize data distribution with appropriate charts and justify the visualization choices.

---

## Part B: Legal Document Classification and Topic Analysis

### B1. Supervised Classification of Legal Texts

Using the Greek Legal Code dataset (from the National and Kapodistrian University of Athens), containing \~47,000 law documents with labels for collection, chapter, and topic, the goal is to build and compare models that classify documents based on their text.

### Models Implemented:

* Support Vector Machines (SVM) with Bag-of-Words and TF-IDF features
* Logistic Regression using dense embeddings (Word2Vec, fastText, or GloVe)
* A third classifier chosen from K-Nearest Neighbors, Naive Bayes, Random Forest, MLP, or XGBoost using either TF-IDF or dense embeddings

### Evaluation:

* Models are trained on the training set, hyperparameters tuned on validation, and performance evaluated on the test set.
* Metrics reported include Accuracy, Precision, Recall, and F1-score.
* Results are presented in tabular format per label category.

---

### B2. Topic Analysis of Supreme Court Decisions

Using the GreekLegalSum dataset (no crawling required here), which contains decision texts, summaries, case categories, and tags, we conduct:

* Exploratory data analysis (EDA) on thematic tags and categories with frequency-based visualizations.
* K-means clustering on textual representations (TF-IDF or embeddings) of decisions or their summaries.
* Selection of optimal number of clusters (K) using Silhouette scores and Normalized Mutual Information (NMI).
* Visualization and interpretation of clustering results.
* Use of a Large Language Model (LLM), specifically Llama-Krikri-8B-Instruct or alternatives, to generate descriptive cluster titles via 3-shot learning.
* Comparison of cluster title generation using decisions chosen randomly versus those closest to the centroid, with argumentation supporting the best method.
* Optional analysis of label distribution per cluster and the relationship between cluster titles and legal tags/categories.

---

## Technologies and Tools

* Python for web scraping and data processing
* NLP libraries for text representation (e.g., scikit-learn, gensim, transformers)
* Machine learning frameworks (scikit-learn, XGBoost, etc.)
* Visualization libraries (matplotlib, seaborn, plotly)
* Large Language Models for topic extraction (Llama-Krikri-8B-Instruct via Unsloth)

---

## Deliverables

* A cleaned and structured dataframe of Supreme Court decisions for 2024
* Comparative evaluation of classification models on Greek legal documents
* Topic clusters and meaningful labels generated for Supreme Court decisions
* Visual and analytical reports documenting data characteristics, model performance, and cluster analysis

