# ---------------------------------------------------------------------
# IMPORTED FUNCTIONS USED IN PROGRAM
# ---------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt


poverty_csv = 'poverty.csv'
education_csv = 'education.csv'
mmr_csv= 'maternal_mortality_rate.csv'

current_poverty_line_value = '$3.00'

# ---------------------------------------------------------------------
# DATA CLEANING SECTION
# -> Including only necessary columns in dataFrames
# -> Renaming columns for esaier analysis
# ---------------------------------------------------------------------

# POVERTY DF
def read_poverty_data():

    df = pd.read_csv(poverty_csv)

    # Mask to get data at 'current_poverty_line_value' 
    poverty_df = df[df['poverty_line'] == current_poverty_line_value ]

    poverty_df =  poverty_df[['year', 'name', 'headcount']].copy()

    poverty_df = poverty_df.rename(columns={
        'year': 'Year',
        'name': 'Country',
        # PR = Poverty Rate (%)
        'headcount': 'PR'
    })
    return poverty_df

poverty_df = read_poverty_data()


# EDUCATION DF
def read_education_data():

    # Deletes first 4 rows of empty data and sets the first row as header
    df = pd.read_csv(education_csv, skiprows=3, header=0)

    df = df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)

    df = df.rename(columns={
        'Country Name': 'Country'
    })

    # Melt the dataframe -> from wide to long
    education_df = df.melt(
        id_vars=['Country'],          # Country column kept the same
        var_name='Year',              # New column for Years
        #PCR = Primary completion rate, total (% of relevant age group)
        value_name='PCR'  # Column for PCR values
    )
    

# MMR DF
def read_MMR_data():

    df = pd.read_csv(mmr_csv)
    MMR_df =  df[['DIM_TIME', 'GEO_NAME_SHORT', 'RATE_PER_100000_N']].copy()

    MMR_df = MMR_df.rename(columns={
        'DIM_TIME': 'Year',
        'GEO_NAME_SHORT': 'Country',
        # MMR = Maternal deaths per 100,000 births
        'RATE_PER_100000_N': 'MMR'
    })
    return  MMR_df

