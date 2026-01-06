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

# ---------------------------------------------------------------------
# TESTING CSVS
# ---------------------------------------------------------------------

    # === Tests if the poverty csv file has been saved/exist ===
    def test_csv_poverty_file_exists(self):
        self.assertTrue(os.path.isfile(poverty_csv_region))

    # === Tests if the education csv file has been saved/exist ===
    def test_csv_education_file_exists(self):
        self.assertTrue(os.path.isfile(education_csv))

    # === Tests if the mmr csv file has been saved/exist ===
    def test_csv_mmr_file_exists(self):
        self.assertTrue(os.path.isfile(mmr_csv))

    # === Tests if the poverty mmr income file has been saved/exist ===
    def test_csv_mmr_income_file_exists(self):    
        self.assertTrue(os.path.isfile(income_classification_csv))
    
# ---------------------------------------------------------------------
# TESTING DATAFRAMES
# ---------------------------------------------------------------------

    # === Tests if the poverty_df has the right columns
    def test_df_poverty_headings(self):
        self.assertEqual(list(poverty_df), ['Country', 'Year', 'PR'])

    # === Tests if the education_df has the right columns
    def test_df_education_headings(self):
        self.assertEqual(list(education_df), ['Country', 'Year', 'PCR'])

    # === Tests if the mmr_df has the right columns
    def test_df_mmr_headings(self):
        self.assertEqual(list(mmr_df), ['Year', 'Country', 'MMR'])

    # === Tests if the mmr_df_income has the right columns
    def test_df_mmr_income_headings(self):
        self.assertEqual(list(mmr_df_income), ['Year', 'Income group', 'Mean_MMR'])

    # === Tests if poverty read_functions returns a dataframe === 
    def test_df_poverty_returned(self):
        self.assertIsInstance(poverty_df, pd.DataFrame)

    # === Tests if education read_functions returns a dataframe === 
    def test_df_education_returned(self):
        self.assertIsInstance(education_df, pd.DataFrame)

    # === Tests if mmr read_functions returns a dataframe === 
    def test_df_mmr_returned(self):
        self.assertIsInstance(mmr_df, pd.DataFrame)

    # === Tests if mmr income read_functions returns a dataframe === 
    def test_df_mmr_income_returned(self):    
        self.assertIsInstance(mmr_df_income, pd.DataFrame)

    # === Checks if all values in Country = 'World' in poverty_df_world ===
    def test_df_world_poverty_only(self):
        poverty_df_world, education_df_world, mmr_df_world = world_filters("Country", "World")
        self.assertTrue((poverty_df_world["Country"] == "World").all())

    # === Checks if all values in Country = 'World' in education_df_world ===
    def test_df_world_education_only(self):
        poverty_df_world, education_df_world, mmr_df_world = world_filters("Country", "World")       
        self.assertTrue((education_df_world["Country"] == "World").all())
 
    # === Checks if all values in Country = 'World' in mmr_df_world ===
    def test_df_world_mmr_only(self):
        poverty_df_world, education_df_world, mmr_df_world = world_filters("Country", "World") 
        self.assertTrue((mmr_df_world["Country"] == "World").all())

# ---------------------------------------------------------------------
# TESTING GRAPHS
# ---------------------------------------------------------------------

# SCATTER PLOT
    # === Checks if function returns a correlation is between -1 to 1 ===
    @patch("matplotlib.pyplot.show")
    def test_scatter_corr_numeric(self, mock_show):
        X_1D, X, merged_df, slope , correlation_coefficient , r_squared_value = plot_scatter_poverty_mmr()
        self.assertTrue(-1 <= correlation_coefficient <= 1)

    # === Checks if function returns a correlation is between 0 to 1 ===
    @patch("matplotlib.pyplot.show")
    def test_scatter_rsquared_numeric(self, mock_show):
        X_1D, X, merged_df, slope , correlation_coefficient , r_squared_value = plot_scatter_poverty_mmr()
        self.assertTrue(0 <= r_squared_value <= 1)

    # === Checks if merged_df merges correctly ===
    @patch("matplotlib.pyplot.show")
    def test_scatter_merged_df(self, mock_show):
        X_1D, X, merged_df, slope , correlation_coefficient , r_squared_value = plot_scatter_poverty_mmr()
        self.assertIn("PR", merged_df.columns)
        self.assertIn("MMR", merged_df.columns)

    # === Checks if X_1D is 1 dimensional ===
    @patch("matplotlib.pyplot.show")
    def test_scatter_X_1D_df(self, mock_show):
        X_1D, X, merged_df, slope , correlation_coefficient , r_squared_value = plot_scatter_poverty_mmr()
        self.assertEqual(X_1D.ndim, 1)

    # === Checks if X_1D is 1 dimensional ===
    @patch("matplotlib.pyplot.show")
    def test_scatter_X_1D_df(self, mock_show):
        X_1D, X, merged_df, slope , correlation_coefficient , r_squared_value = plot_scatter_poverty_mmr()
        self.assertEqual(X_1D.ndim, 1)

    # === Checks if X is 2 dimensional ===
    @patch("matplotlib.pyplot.show")
    def test_scatter_X_df(self, mock_show):
        X_1D, X, merged_df, slope , correlation_coefficient , r_squared_value = plot_scatter_poverty_mmr()
        self.assertEqual(X.ndim, 2)

# BOX PLOT
    # === Tests if years in merged_df only contains target year 2023 ===
    @patch("matplotlib.pyplot.show")
    def test_box_plot_2023_merged_df(self, mock_show):
        merged_df = box_plots()
        self.assertTrue((merged_df["Year"] == 2023).all())


    # === Tests if df merged correctly ===
    @patch("matplotlib.pyplot.show")
    def test_box_plot_merged_df(self, mock_show):
        merged_df = box_plots()  
        self.assertIn("MMR", merged_df.columns)
        self.assertIn("Income group", merged_df.columns)   

# GLOBAL MMR SERIES

    # ===  Checks world_change returned in global/nigeria mmr series is numeric ===
    @patch("matplotlib.pyplot.show")
    def test_mmr_series_world(self, mock_show):
        world_change, nigeria_change = plot_world_nigeria_timeseries()
        self.assertIsInstance(world_change, (int, float))

    # ===  Checks nigeria_change returned in global/nigeria mmr series is numeric ===
    @patch("matplotlib.pyplot.show")
    def test_mmr_series_nigeria(self, mock_show):
        world_change, nigeria_change = plot_world_nigeria_timeseries()
        self.assertIsInstance(nigeria_change, (int, float))

    
        
# INCOME GROUP SCATTER

    # === Testing if df merged correctly ===
    @patch("matplotlib.pyplot.show")
    def test_income_group_scatter_merged_df(self, mock_show):
        income_group = "Low income"
        merged_df, r_value, r_squared_value = plot_income_group_scatter(income_group)
        self.assertIn("Mean_MMR", merged_df.columns)
        self.assertIn("Income group", merged_df.columns)   

    # === Checks if function returns a correlation is between -1 to 1 ===
    @patch("matplotlib.pyplot.show")
    def test_income_group_scatter_corr_numeric(self, mock_show):
        income_group = "Low income"
        merged_df, r_value, r_squared_value = plot_income_group_scatter(income_group)
        self.assertTrue(-1 <= r_value <= 1)

    # === Checks if function returns a correlation is between 0 to 1 ===
    @patch("matplotlib.pyplot.show")
    def test_income_group_scatter_rsquared_numeric(self, mock_show):
        income_group = "Low income"
        merged_df, r_value, r_squared_value = plot_income_group_scatter(income_group)
        self.assertTrue(0 <= r_squared_value <= 1)


# ---------------------------------------------------------------------
# TESTING VALUES 
# ---------------------------------------------------------------------
    # === Checks if function returns a numeric value or float for 2023 mmr value ===
    def test_get_global_mmr_returns_numeric_mmr(self):
        mmr_2023, percentage_lmic = get_global_mmr_values()
        self.assertIsInstance(mmr_2023, (int, float))

    # === Checks if function returns a numeric value or float for percentage ===
    def test_get_global_mmr_returns_numeric_percentage_lmic(self):
        mmr_2023, percentage_lmic = get_global_mmr_values()
        self.assertIsInstance(percentage_lmic, (int, float))

    # === Checks if value returned is above 0 ===
    def test_get_global_mmr_value_range_mmr(self):
        mmr_2023, percentage_lmic = get_global_mmr_values()
        self.assertGreaterEqual(mmr_2023, 0)

    # === Checks if function returns a percentage between 0 to 100 ===
    def test_get_global_mmr_value_percentage_lmic(self):
        mmr_2023, percentage_lmic = get_global_mmr_values()
        self.assertTrue(0 <= percentage_lmic <= 100)

    # === Checks if function returns a numeric value or float for poverty ===
    def test_get_global_mmr_returns_numeric_percentage_poverty_pop(self):
        pov_2023_value = percentage_pop_poverty()
        self.assertIsInstance(pov_2023_value, (int, float))

    # === Checks if value returned if value returned is between 0 to 100 ===
    def test_get_global_mmr_value_range_poverty_pop(self):
        pov_2023_value =  percentage_pop_poverty()
        self.assertTrue(0 <= pov_2023_value <= 100)

    # run the tests
if __name__ == "__main__":
    unittest.main()