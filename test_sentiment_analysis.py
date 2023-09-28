import unittest
from sentiment_analysis import train_model, test_model, plot_wordclouds
import nltk

class TestMethods(unittest.TestCase):

    def setUp(self):
        nltk.download('movie_reviews')
        self.example_data = ['This is a positive review', 'This is a negative review']
        self.example_labels = ['pos', 'neg']

    def test_train_model(self):
        model = train_model(self.example_data, self.example_labels)
        self.assertIsNotNone(model)

    def test_test_model(self):
        model = train_model(self.example_data, self.example_labels)
        predictions = test_model(self.example_data, self.example_labels, model)
        self.assertEqual(len(predictions), 2)

    def test_plot_wordclouds(self):
        try:
            nltk.download('stopwords')
            plot_wordclouds(self.example_data, self.example_labels)
            self.assertTrue(True)  # Passed if no error occurs
        except Exception as e:
            self.fail(f"create_wordcloud() raised {type(e)} with message {e}")

if __name__ == '__main__':
    unittest.main()
    
