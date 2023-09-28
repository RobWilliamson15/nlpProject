"""
Loads and runs sentiment analysis on the nltk dataset movie_reviews
"""
import random
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Imports for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def get_data():
    """
    This function downloads, loads and splits the data ready for training and testing
    """
    nltk.download('movie_reviews')
    documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    # Convert the list of words in each review to a single string
    data = [' '.join(words) for words, label in documents]
    labels = [label for words, label in documents]

    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train):
    """
    This function trains the model
    """
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(x_train, y_train)

    return model

def test_model(x_test, y_test, model):
    """
    This funciton runs a prediction on the test data
    """
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    return y_pred

def plot_confusion_matrix(y_test, y_pred):
    """
    This function plots a confusion matrix
    """
    c_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(c_matrix, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def plot_wordclouds(x_train, y_train):
    """
    This function plots word clouds for both positive and negative reviews
    """
    positive_reviews = ' '.join([review for review, label in
        zip(x_train, y_train) if label == 'pos'])

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

    negative_reviews = ' '.join([review for review, label in
        zip(x_train, y_train) if label == 'neg'])

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

def plot_lengths(x_train):
    """
    This function plots a distribution of review length
    """
    review_lengths = [len(review.split()) for review in x_train]

    plt.figure(figsize=(10,6))
    plt.hist(review_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

if __name__ == '__main__':
    x_train_outer, x_test_outer, y_train_outer, y_test_outer = get_data()
    model_outer = train_model(x_train_outer, y_train_outer)
    y_pred_outer = test_model(x_test_outer, y_test_outer, model_outer)
    plot_confusion_matrix(y_test_outer, y_pred_outer)
    plot_wordclouds(x_train_outer, y_train_outer)
    plot_lengths(x_train_outer)
