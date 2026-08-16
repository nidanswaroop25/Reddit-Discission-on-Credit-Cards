# Credit Card Reddit Discussion Classification & Topic Discovery

##  Project Overview

This project analyzes **Reddit discussions related to credit cards** using Natural Language Processing (NLP), Machine Learning, and Unsupervised Learning techniques.

The project has two major components:

1. **Supervised Text Classification** – classifying Reddit comments into predefined categories using multiple machine-learning algorithms.
2. **Unsupervised Topic Discovery** – discovering meaningful discussion themes from Reddit comments using clustering and sentence embeddings.

The project compares traditional **TF-IDF-based machine-learning models** with **sentence-embedding-based approaches** to understand patterns in credit-card discussions.

---

##  Objectives

* Perform comprehensive text preprocessing on Reddit discussions.
* Convert unstructured text into numerical features using **TF-IDF**.
* Build and compare multiple supervised classification models.
* Address class imbalance in the target categories.
* Identify hidden discussion patterns using clustering.
* Generate semantic representations using **Sentence Transformers**.
* Visualize discussion clusters using **UMAP**.
* Extract representative keywords and interpret the discovered clusters.
* Evaluate whether semantic embeddings can be used effectively for supervised classification.

---

##  Dataset

The project uses a labeled Reddit dataset containing **credit-card-related discussions**.

Each observation represents a Reddit comment/discussion along with its corresponding category label.

The dataset is loaded from:

```text
labeled(4)_reddit_credit_card_discussion.csv
```

The primary text variable is the Reddit **comment**, while the `category` variable is used as the supervised-learning target.

---

#  Text Preprocessing

Extensive preprocessing was performed to convert raw Reddit comments into clean textual data.

The preprocessing pipeline included:

### 1. Lowercasing

Converted all text to lowercase to ensure consistency.

### 2. Emoji Removal

Removed emojis from the comments.

### 3. Number Removal

Removed numerical characters where appropriate.

### 4. Punctuation Removal

Removed punctuation marks that were not useful for the classification task.

### 5. Removing Line Breaks, Hyphens & Extra Spaces

Cleaned formatting-related noise from Reddit comments.

### 6. Chat Word Expansion

Expanded commonly used chat abbreviations into their corresponding words.

### 7. Language Detection

Identified the language of comments to improve the quality of subsequent NLP processing.

### 8. Tokenization

Split comments into individual tokens/words.

### 9. POS Tagging & Lemmatization

Part-of-speech tagging and lemmatization were performed to reduce words to their meaningful base forms.

### 10. Stop Word Removal

Common words with limited information content were removed.

---

# Train-Test Split

The dataset was divided into training and testing sets using:

```python
train_test_split()
```

A stratified split was used in the embedding-based classification stage to preserve the distribution of the target classes.

---

# Text Representation

## TF-IDF

The primary text representation used for traditional machine-learning models was **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The main configuration used:

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True
)
```

This represents each comment using:

* Unigrams
* Bigrams
* Up to 20,000 features

TF-IDF was used as input for the supervised classification models.

---

# 🤖 Supervised Classification

Multiple machine-learning algorithms were trained and compared.

## 1. Logistic Regression

A Logistic Regression classifier was trained using TF-IDF features.

```python
LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
```

---

## 2. Linear SVM

A Linear Support Vector Machine was implemented using:

```python
LinearSVC(
    class_weight='balanced'
)
```

The model was evaluated using classification reports and confusion matrices.

---

## 3. Multinomial Naïve Bayes

Multinomial Naïve Bayes was applied because of its suitability for text classification with non-negative TF-IDF features.

---

## 4. Random Forest

A Random Forest classifier was implemented with:

```python
n_estimators=300
class_weight='balanced'
random_state=42
```

---

## 5. XGBoost

An XGBoost classifier was also trained using the TF-IDF representation.

Key parameters included:

```text
n_estimators = 100
learning_rate = 0.1
max_depth = 3
subsample = 0.8
colsample_bytree = 0.8
```

---

## 6. LightGBM

A LightGBM classifier was trained with class weighting to account for class imbalance.

Key parameters included:

```text
n_estimators = 300
learning_rate = 0.1
class_weight = balanced
```

---

# Handling Class Imbalance

Class imbalance was investigated using two approaches.

### 1. Class Weighting

Models were trained using:

```python
class_weight='balanced'
```

The analysis showed that simply changing class weights did not improve the results and could degrade performance.

### 2. Random Oversampling

Random oversampling was applied to the training data using:

```python
RandomOverSampler(random_state=42)
```

The resampled training data was then used to train Logistic Regression and Linear SVM models.

Importantly, oversampling was applied to the **training data**, while the original test set was retained for evaluation.

---

#  Model Evaluation

The supervised models were evaluated using:

* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

The classification report provides class-wise performance and allows comparison of the models across the different discussion categories.

---

#  Unsupervised Learning & Clustering

In addition to supervised classification, the project explored the underlying structure of Reddit discussions without relying on predefined labels.

---

## K-Means Clustering

TF-IDF representations were used as input to K-Means clustering.

The analysis tested different numbers of clusters and identified:

```text
k = 4
```

as the preferred number of clusters within the investigated range.

Cluster quality was evaluated using:

* Silhouette Score
* Calinski-Harabasz Index
* Davies-Bouldin Index

---

## Agglomerative Clustering

Agglomerative hierarchical clustering was also explored.

Dimensionality reduction using **Truncated SVD** was performed before clustering:

```python
TruncatedSVD(
    n_components=100,
    random_state=42
)
```

---

#  Sentence Embeddings

To capture the semantic meaning of Reddit comments beyond TF-IDF word frequencies, the project used a **Sentence Transformer** model:

```text
all-MiniLM-L6-v2
```

Each Reddit comment was transformed into a dense semantic embedding.

These embeddings were subsequently used for clustering and supervised classification.

---

# UMAP + HDBSCAN

The sentence embeddings were further analyzed using dimensionality reduction and density-based clustering.

### UMAP

UMAP was used to reduce the dimensionality of the sentence embeddings.

Configuration included:

```text
n_neighbors = 15
n_components = 5
metric = cosine
```

### HDBSCAN

HDBSCAN was used to identify groups of semantically similar Reddit discussions.

This approach allows clusters to be identified based on the semantic structure of the comments rather than only word-level similarity.

---

# Semantic Cluster Interpretation

The discovered clusters were interpreted using their most representative keywords.

The four major discussion themes identified were:

### 1. Account & Spending Concerns

Representative keywords included:

```text
month, limit, salary, pay, EMI
```

This cluster represents discussions related to spending, monthly payments, credit limits, salaries, and EMIs.

### 2. Credit Approval Issues

Representative keywords included:

```text
card, credit, SBI, CIBIL, reject
```

This cluster represents discussions around credit-card applications, approval/rejection, CIBIL scores, and eligibility.

### 3. General Positive Feedback

Representative keywords included:

```text
thanks, yes, work, use
```

This cluster contains generally positive or supportive discussions.

### 4. Offers & Rewards

Representative keywords included:

```text
HDFC, cashback, rewards, LTF
```

This cluster represents discussions about credit-card offers, cashback, rewards, and lifetime-free cards.

---

#  Cluster Visualization

UMAP was used to visualize the semantic clusters in a lower-dimensional space.

The visualization provides an intuitive representation of how Reddit discussions group together according to their semantic content.

---

#  Embedding-Based Supervised Learning

The sentence embeddings generated using `all-MiniLM-L6-v2` were also used as features for supervised classification.

Two classifiers were trained:

### Logistic Regression

A multinomial Logistic Regression classifier was trained using the sentence embeddings.

### Random Forest

A Random Forest classifier was also trained with:

```text
n_estimators = 200
class_weight = balanced
random_state = 42
```

Both models were evaluated using:

* Classification Report
* Confusion Matrix

This provides a comparison between classification using traditional TF-IDF features and classification using semantic sentence embeddings.

---

# Technologies & Libraries

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### NLP

* NLTK
* Text preprocessing utilities
* Sentence Transformers

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* Imbalanced-learn

### Clustering

* K-Means
* Agglomerative Clustering
* HDBSCAN

### Dimensionality Reduction

* Truncated SVD
* UMAP

### Visualization

* Matplotlib
* Seaborn

---

# Project Structure

```text
Credit-Card-Reddit-Discussion-Classification/
│
├── Credit_Card_Reddit_Discussion_Classification.ipynb
├── labeled(4)_reddit_credit_card_discussion.csv
└── README.md
```

---

#  Key Findings

* Extensive NLP preprocessing was required to clean noisy Reddit discussions.
* **TF-IDF with unigram and bigram features** provided a strong representation for traditional machine-learning models.
* Multiple classification algorithms were compared, including Logistic Regression, Linear SVM, Naïve Bayes, Random Forest, XGBoost, and LightGBM.
* Class imbalance was investigated using both **class weighting and random oversampling**.
* Unsupervised clustering identified **four major discussion themes**.
* Sentence Transformers provided a semantic representation of the Reddit comments.
* **UMAP and clustering** helped visualize and interpret the underlying semantic structure.
* Sentence embeddings were additionally used as features for supervised classification.

---

#  Future Improvements

Possible extensions include:

* Hyperparameter tuning using GridSearchCV or Bayesian optimization.
* Testing transformer-based classifiers such as BERT/RoBERTa.
* Fine-tuning a transformer specifically on credit-card discussions.
* Comparing TF-IDF, Word2Vec and transformer embeddings systematically.
* Using cross-validation for more robust model comparison.
* Developing an automated Reddit discussion monitoring system.
* Performing sentiment analysis alongside topic classification.
* Building a dashboard for tracking emerging credit-card discussion topics.
* Applying topic modelling techniques such as BERTopic.

---

