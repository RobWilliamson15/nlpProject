"""
This performs unit tests for sentiment_analysis
"""
import unittest
import nltk
from sentiment_analysis import train_model, test_model, plot_wordclouds

class TestMethods(unittest.TestCase):
    def setUp(self):
        """
        Setup up data for unit tests
        """
        nltk.download('movie_reviews')
        self.example_data = ['This is a positive review', 'This is a negative review']
        self.example_labels = ['pos', 'neg']

    def test_train_model(self):
        """
        Test the training of the model
        """
        model = train_model(self.example_data, self.example_labels)
        self.assertIsNotNone(model)

    def test_test_model(self):
        """
        Test the predictions of the model
        """
        model = train_model(self.example_data, self.example_labels)
        predictions = test_model(self.example_data, self.example_labels, model)
        self.assertEqual(len(predictions), 2)

    def test_plot_wordclouds(self):
        """
        Test if the wordclouds plot can be made
        """
        try:
            nltk.download('stopwords')
            plot_wordclouds(self.example_data, self.example_labels)
            self.assertTrue(True)  # Passed if no error occurs
        except Exception as failed_plot:
            self.fail(f"create_wordcloud() raised {type(failed_plot)} with message {failed_plot}")

if __name__ == '__main__':
    unittest.main()
