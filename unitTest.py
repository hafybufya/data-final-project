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

        self.assertTrue(os.path.isfile('poverty_country.csv'))
        self.assertTrue(os.path.isfile('poverty_region.csv'))
        self.assertTrue(os.path.isfile('education.csv'))
        self.assertTrue(os.path.isfile('maternal_mortality_rate.csv'))

    # === Tests if read_functions returns dataframes === 
    def test_df_returned(self):
        # From read_poverty_data() function
        self.assertTrue( isinstance(poverty_df, pd.DataFrame))
        # From read_education_data() function
        self.assertTrue( isinstance(education_df, pd.DataFrame))
        # From read_MMR_data() function
        self.assertTrue( isinstance(MMR_df, pd.DataFrame))

    # run the tests
if __name__ == "__main__":
    unittest.main()