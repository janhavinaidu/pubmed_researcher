Author MeSH Profile Generator

This is a machine learning mini-project that trains a multi-label classifier to predict broad topic categories for medical abstracts from PubMed. The trained model is deployed in a Streamlit web application that can generate a "thematic profile" for any given author based on their publication history.

1. Problem Statement

PubMed, a primary database for biomedical literature, indexes articles with highly specific MeSH (Medical Subject Headings). With over 29,000 unique MeSH terms, this creates two major problems:

Label Sparsity: Many labels are extremely rare, making it difficult for a machine learning model to learn their patterns.

Information Overload: It's difficult for a human to quickly assess the broad thematic focus of an author or a search query when faced with hundreds of hyper-specific terms.

This project solves this by implementing a hierarchical classification approach. Instead of predicting from 29,000+ labels, we train models to predict only the 14 broad, top-level MeSH root categories (e.g., 'A' for Anatomy, 'C' for Diseases, 'D' for Chemicals and Drugs).

2. Methodology & ML Pipeline

The project followed a standard machine learning workflow, with a key constraint of using only basic classification algorithms and avoiding heavy libraries like NLTK.

2.1. Data Preprocessing & EDA

Dataset: Used the "PubMed Multi Label Text Classification Dataset Processed.csv".

Cleaning: Removed missing abstracts and duplicates. Performed text cleaning using basic Python re (regex) to lowercase text and remove punctuation/numbers.

Feature Creation: Created text_length and label_count columns for analysis.

EDA (Key Finding): A clustered heatmap of the 14 root labels showed strong positive correlations between categories (e.g., 'Diseases' & 'Chemicals/Drugs', 'Anatomy' & 'Diseases'). This data-driven insight proved that the labels are not independent.

2.2. Feature Engineering

Technique: TfidfVectorizer (from scikit-learn).

Parameters:

max_features=5000: Kept the model lightweight by focusing on the top 5,000 words.

stop_words='english': Used the vectorizer's built-in stop word list as a simple, NLTK-free alternative.

The cleaned training text was fit_transform-ed, and the test text was transform-ed.

2.3. Model Development & Hyperparameter Tuning

Based on the EDA, the ClassifierChain (CC) strategy was chosen as the main model to leverage the observed label dependencies. This was compared against other basic classifiers.

Models Compared:

ClassifierChain(LogisticRegression())

ClassifierChain(MultinomialNB())

ClassifierChain(SGDClassifier())

Tuning: GridSearchCV (with 3-fold cross-validation, scoring f1_weighted) was used to tune the key hyperparameter for each model:

Logistic Regression: Tuned C (regularization strength). Best: 1

Naive Bayes: Tuned alpha (smoothing). Best: 0.1

SGD Classifier: Tuned alpha (regularization). Best: 0.0001

3. Results & Model Selection

The models were evaluated on the unseen test set. The ClassifierChain + Logistic Regression model was the clear winner.

Final Model Comparison

Model

Exact Match

Hamming Loss

Precision (Weighted)

Recall (Weighted)

F1 (Weighted)

Logistic Regression

0.1550

0.1295

0.8456

0.8258

0.8267

SGD Classifier (Log Loss)

0.1451

0.1346

0.8439

0.8145

0.8126

Naive Bayes

0.1107

0.1580

0.8048

0.8138

0.8068

Conclusion: The ClassifierChain with Logistic Regression was selected as the final model. It achieved the lowest Hamming Loss (12.95%) and the highest Weighted F1-Score (82.7%), proving it provided the best balance of precision and recall.

Key Insight: The detailed report showed the model excels at common categories (like 'B: Organisms') but struggles with rare ones (like 'H: Disciplines') due to class imbalance, which is a key area for future improvement.

4. Deployed Application

The final trained Logistic Regression model and TfidfVectorizer were saved as .joblib files and deployed in a Streamlit web application.

App Link: https://pubmedresearchergit-mov3njrga3zwvqclpyhkkb.streamlit.app/

Functionality:

User enters an author's name in PubMed format (e.g., Fauci AS[AU]).

The app queries the PubMed API, filters for articles with abstracts, and fetches the details.

It classifies each abstract in real-time using the saved model.

It displays an Analytics Dashboard with the author's aggregate "MeSH Root Profile," including bar charts and a list of classified papers.

5. Technology Stack

Language: Python

Machine Learning: Scikit-learn (TfidfVectorizer, ClassifierChain, LogisticRegression, GridSearchCV, classification_report)

Data Handling: Pandas, NumPy

Web App: Streamlit

Data Fetching: requests (for PubMed API), xml.etree.ElementTree

Plotting: Plotly Express

How to Run This Project Locally

Clone the repository:

git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>


Create a virtual environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required libraries:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py
