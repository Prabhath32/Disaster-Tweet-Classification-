import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st

nltk.download('stopwords')
nltk.download('punkt')

# Step 1: Load Dataset
def load_dataset():
    try:
        df = pd.read_csv("train.csv")  # Use your actual file path if different
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'train.csv' is in the same directory.")
        st.stop()
    return df

# Step 2: Data Cleaning
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtag symbol
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Step 3: EDA
def eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")
    
    # Class distribution
    st.write("Class Distribution")
    plt.figure(figsize=(5, 3))
    sns.countplot(x=df['target'])
    st.pyplot(plt.gcf())
    plt.clf()
    
    # Tweet length distribution
    df['tweet_length'] = df['text'].apply(len)
    st.write("Tweet Length Distribution")
    plt.figure(figsize=(5, 3))
    sns.histplot(df['tweet_length'], bins=30, kde=True)
    st.pyplot(plt.gcf())
    plt.clf()

    # Word cloud for disaster tweets
    if 'cleaned_text' in df.columns:
        disaster_tweets = ' '.join(df[df['target'] == 1]['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(disaster_tweets)
        st.write("Word Cloud for Disaster-Related Tweets")
        st.image(wordcloud.to_array())

# Step 4: Model Training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model evaluation
    st.subheader("Model Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))
    return model

# Step 5: Predict on Test Dataset
def predict_test_data(model, vectorizer):
    try:
        test_df = pd.read_csv("test.csv")
        test_df['cleaned_text'] = test_df['text'].apply(clean_text)
        X_test_tfidf = vectorizer.transform(test_df['cleaned_text'])
        test_predictions = model.predict(X_test_tfidf)
        
        output = pd.DataFrame({
            'id': test_df['id'],
            'target': test_predictions
        })
        output.to_csv('submission.csv', index=False)
        st.subheader("Predictions on Test Data")
        st.write("Predictions saved successfully as 'submission.csv'")
        st.dataframe(output.head())
    except FileNotFoundError:
        st.warning("Test dataset not found. Please ensure 'test.csv' is in the same directory.")

# Main App
stop_words = set(stopwords.words('english'))

def main():
    st.title("Disaster Tweet Classification")
    
    # Load Dataset
    df = load_dataset()
    st.write("Dataset Loaded")
    st.dataframe(df.head())

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    df['cleaned_text'] = df['text'].apply(clean_text)
    st.write("Cleaned Data Sample")
    st.dataframe(df[['text', 'cleaned_text']].head())

    # EDA
    eda(df)

    # Vectorization
    st.subheader("Vectorizing Text Data")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['target']
    st.write("TF-IDF Vectorization Completed")

    # Model Training
    model = train_model(X, y)

    # Predict on Test Data
    predict_test_data(model, vectorizer)

if __name__ == '__main__':
    main()
