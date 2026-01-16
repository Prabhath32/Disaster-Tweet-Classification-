# Disaster-Tweet-Classification

---

#  Disaster Tweet Classification using Machine Learning

##  Project Overview

This project builds a **machine learning–based text classification system** to identify whether a tweet is related to a real disaster or not.
It uses **Natural Language Processing (NLP)** techniques and a **Logistic Regression model**, deployed through a **Streamlit web application**.


---

##  Project Structure

```
├── app.py               # Streamlit application
├── train.csv            # Training dataset
├── test.csv             # Test dataset
├── submission.csv       # Model predictions
├── README.md            # Project documentation
```

---

##  Dataset Description

###  train.csv

* **text** → Tweet content
* **target** →

  * `1` = Disaster-related tweet
  * `0` = Not disaster-related

###  test.csv

* **id** → Tweet ID
* **text** → Tweet content

###  submission.csv

* **id** → Tweet ID
* **target** → Model prediction

---

##  Machine Learning Workflow

### 1️ Data Loading

* Dataset loaded using **Pandas**
* Error handling included if files are missing

---

### 2️ Text Preprocessing

Each tweet is cleaned by:

* Removing URLs, mentions, hashtags
* Removing special characters and numbers
* Converting text to lowercase
* Removing stopwords using **NLTK**

---

### 3️ Exploratory Data Analysis (EDA)

* Class distribution visualization
* Tweet length distribution
* Word Cloud for disaster-related tweets

---

### 4️ Feature Extraction

* **TF-IDF Vectorization**
* Maximum features set to **5000**

---

### 5️ Model Building

* **Algorithm:** Logistic Regression
* **Train-Test Split:** 80:20
* **Max Iterations:** 500

---

### 6️ Model Evaluation

Evaluation metrics used:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report

---

### 7️ Prediction on Test Data

* Model predicts disaster relevance on `test.csv`
* Results saved automatically as:

```
submission.csv
```

---

##  Streamlit Web Application

The Streamlit app provides:

* Dataset preview
* Cleaned text visualization
* Interactive EDA plots
* Model training & evaluation
* Automatic prediction generation

To run the app:

```bash
streamlit run app.py
```

---

##  Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* NLTK
* Streamlit

---

##  Key Highlights

* End-to-end NLP pipeline
* Real-time visualizations
* Automated prediction export
* Beginner-friendly & interview-ready project

---

##  Conclusion

This project demonstrates a **complete NLP-based classification system**, combining data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

---
