import nltk
import random
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Imports for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

import unittest

def get_data():
    nltk.download('movie_reviews')
    documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    # Convert the list of words in each review to a single string
    data = [' '.join(words) for words, label in documents]
    labels = [label for words, label in documents]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    return model

def test_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');

def plot_wordclouds(X_train, y_train):
    positive_reviews = ' '.join([review for review, label in zip(X_train, y_train) if label == 'pos'])

    wordcloud = WordCloud(width = 800, height = 800, 
                      background_color ='white', 
                      stopwords = set(nltk.corpus.stopwords.words('english')), 
                      min_font_size = 10).generate(positive_reviews)

    plt.figure(figsize=(8,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Word Cloud for Positive Reviews")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    negative_reviews = ' '.join([review for review, label in zip(X_train, y_train) if label == 'neg'])

    wordcloud = WordCloud(width = 800, height = 800, 
                      background_color ='white', 
                      stopwords = set(nltk.corpus.stopwords.words('english')), 
                      min_font_size = 10).generate(negative_reviews)

    plt.figure(figsize=(8,8))
    plt.imshow(wordcloud)   
    plt.axis("off")
    plt.title("Word Cloud for Negative Reviews")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def plot_lengths(X_train):
    review_lengths = [len(review.split()) for review in X_train]

    plt.figure(figsize=(10,6))
    plt.hist(review_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    model = train_model(X_train, y_train)
    y_pred = test_model(X_test, y_test, model)
    plot_confusion_matrix(y_test, y_pred)
    plot_wordclouds(X_train, y_train)
    plot_lengths(X_train)
