# ---------------------------------------------------------------------
# IMPORTED FUNCTIONS USED IN PROGRAM
# ---------------------------------------------------------------------

import unittest
from unittest.mock import patch
from mainCode import *
import os

# ---------------------------------------------------------------------
# Class of unit tests
# ---------------------------------------------------------------------
class my_unit_tests(unittest.TestCase):

    # === Tests if the csvs file has been saved/exist ===
    def test_csv_file_exists(self):
        self.assertTrue(os.path.isfile(poverty_csv_region))
        self.assertTrue(os.path.isfile(education_csv))
        self.assertTrue(os.path.isfile(mmr_csv))
        self.assertTrue(os.path.isfile(income_classification_csv))
    
    # === Tests if the poverty_df has the right columns
    def test_df_poverty_headings(self):
        self.assertEqual(list(poverty_df), ['Country', 'Year', 'PR'])

    # === Tests if the education_df has the right columns
    def test_df_education_headings(self):
        self.assertEqual(list(education_df), ['Country', 'Year', 'PCR'])

    # === Tests if the MMR_df has the right columns
    def test_df_MMR_headings(self):
        self.assertEqual(list(MMR_df), ['Year', 'Country', 'MMR'])

    # === Tests if the MMR_df_income has the right columns
    def test_df_MMR_income_headings(self):
        self.assertEqual(list(MMR_df_income), ['Year', 'Income group', 'Mean_MMR'])


    # === Tests if read_functions returns dataframes === 
    def test_df_returned(self):
        # From read_poverty_data() function
        self.assertTrue( isinstance(poverty_df, pd.DataFrame))
        # From read_education_data() function
        self.assertTrue( isinstance(education_df, pd.DataFrame))
        # From read_MMR_data() function
        self.assertTrue( isinstance(MMR_df, pd.DataFrame))
        self.assertTrue( isinstance(MMR_df_income, pd.DataFrame))

    # === Checks if function returns a correlation and r squared value between -1 to 1 ===
    @patch("matplotlib.pyplot.show")
    def test_scatter_corr_numeric(self, mock_show):
        slope , correlation_coefficient , r_squared_value = plot_scatter_poverty_MMR()
        self.assertTrue(-1 <= correlation_coefficient <= 1)
        self.assertTrue(-1 <= r_squared_value <= 1)

    # === Checks if function returns a correlation and r squared value between -1 to 1 ===
    def test_df_world_only(self):
        poverty_df_world, education_df_world, MMR_df_world = world_filters()
        self.assertTrue((poverty_df_world["Country"] == "World").all())
        self.assertTrue((education_df_world["Country"] == "World").all())
        self.assertTrue((MMR_df_world["Country"] == "World").all())

    # === Checks if function returns a correlation and r squared value between -1 to 1 ===
    def test_df_world_only(self):
        poverty_df_world, education_df_world, MMR_df_world = world_filters()
        self.assertTrue((poverty_df_world["Country"] == "World").all())
        self.assertTrue((education_df_world["Country"] == "World").all())
        self.assertTrue((MMR_df_world["Country"] == "World").all())
     
    # run the tests
if __name__ == "__main__":
    unittest.main()