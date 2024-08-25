import unittest
import pandas as pd
from src.drawfinal import plot_top_tfidf_words

class TestDrawFinal(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        data = {
            'Title': ['Title1', 'Title2'],
            'Columns': ['Column1', 'Column2'],
            'Tags': ['[Tag1;Tag2]', '[Tag3;Tag4]'],
            'Abstract': ['Abstract1', 'Abstract2']
        }
        self.df = pd.DataFrame(data)
        self.csv_file = 'test_data.csv'
        self.df.to_csv(self.csv_file, index=False)

    def test_plot_top_tfidf_words(self):
        # Test if the function runs without errors
        try:
            plot_top_tfidf_words(self.csv_file)
        except Exception as e:
            self.fail(f"plot_top_tfidf_words raised an exception: {e}")

    def tearDown(self):
        # Delete file after testing
        import os
        os.remove(self.csv_file)

if __name__ == '__main__':
    unittest.main()