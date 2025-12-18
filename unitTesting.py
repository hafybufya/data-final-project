# ---------------------------------------------------------------------
# IMPORTED FUNCTIONS USED IN PROGRAM
# ---------------------------------------------------------------------

import unittest
from mainCode import *
import os

# ---------------------------------------------------------------------
# Class of unit tests
# ---------------------------------------------------------------------
class my_unit_tests(unittest.TestCase):

    # === Tests if the csvs file has been saved/exist ===
    def test_csv_file_exists(self):

        self.assertTrue(os.path.isfile(poverty_csv_country))
        self.assertTrue(os.path.isfile(poverty_csv_world))
        self.assertTrue(os.path.isfile(education_csv))
        self.assertTrue(os.path.isfile(mmr_csv))
        self.assertTrue(os.path.isfile(income_classification_csv))
      

    # === Tests if read_functions returns dataframes === 
    def test_df_returned(self):
        # From read_poverty_data() function
        self.assertTrue( isinstance(poverty_df, pd.DataFrame))
        # From read_education_data() function
        self.assertTrue( isinstance(education_df, pd.DataFrame))
        # From read_MMR_data() function
        self.assertTrue( isinstance(MMR_df, pd.DataFrame))
        self.assertTrue( isinstance(MMR_df_income, pd.DataFrame))

    # Checks if functions returns a value between -1 to 1
    def test_scatter_corr_numeric(self):
        result = plot_scatter_poverty_MMR()
        self.assertTrue(-1 <= result <= 1)
        
    # run the tests
if __name__ == "__main__":
    unittest.main()