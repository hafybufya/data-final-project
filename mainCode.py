# ---------------------------------------------------------------------
# IMPORTED FUNCTIONS USED IN PROGRAM
# ---------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# CSVs used in programs defined
csv_folder = 'data'
poverty_csv_country = f'{csv_folder}/poverty_country.csv'
poverty_csv_world = f'{csv_folder}/poverty_region.csv'
education_csv = f'{csv_folder}/education.csv'
mmr_csv= f'{csv_folder}/maternal_mortality_rate.csv'
income_classification_csv = f'{csv_folder}/income_group_classification.csv'

# ---------------------------------------------------------------------
# DATA CLEANING SECTION
# -> Including only necessary columns in dataFrames
# -> Renaming columns for esaier analysis
# ---------------------------------------------------------------------

# POVERTY WORlD DF 

def read_poverty_data():

    poverty_df = pd.read_csv(poverty_csv_world)
    poverty_df =  poverty_df[['region_name', 'reporting_year', 'headcount']].copy()

    poverty_df = poverty_df.rename(columns={
        'reporting_year': 'Year',
        'region_name': 'Country',
        # PR = Poverty Rate (%)
        'headcount': 'PR'
    })
    return poverty_df


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
        # PCR = Primary completion rate, total (% of relevant age group)
        value_name='PCR'  # Column for PCR values
    )
    # education_df.to_csv('filename.csv', index=False)

    return education_df

# MMR DF
def read_MMR_data():

    df = pd.read_csv(mmr_csv)
    MMR_df =  df[['DIM_TIME', 'GEO_NAME_SHORT',  'RATE_PER_100000_N']].copy()

    MMR_df = MMR_df.rename(columns={
        'DIM_TIME': 'Year',
        'GEO_NAME_SHORT': 'Country',
        # MMR = Maternal deaths per 100,000 births
        'RATE_PER_100000_N': 'MMR'
    })
    return  MMR_df

def read_MMR_income_data():
    #Columns: Year, Income group, Mean MMR 
    
    df = pd.read_csv(income_classification_csv)
    income_df =  df[['Economy', 'Income group']].copy()
    income_df = income_df.rename(columns={
        # Same name for merging with MMR df
        'Economy': 'Country'})
    
    # Merge Datasets
    merged_df = pd.merge(MMR_df , income_df, on=["Country"] )
    
    result = (
        merged_df
        # as_index = False ensures grouping stays as normal columsn
        .groupby(["Year", "Income group"], as_index=False).agg(Mean_MMR=("MMR", "mean"))
    )
    result.to_csv('filename.csv', index=False)


    return result


# ---------------------------------------------------------------------
# Hypothesis 1
# Do poorer countries have higher maternal mortality rates?
# ---------------------------------------------------------------------

#Future Improvement: Adding a heatmap for easier visualisation

def plot_scatter_poverty_MMR():


    # Plotting both dfs told only include 'World' entity
    poverty_df_world = poverty_df[poverty_df["Country"] == "World"]
    MMR_df_world = MMR_df[MMR_df["Country"] == "World"]


    # Merge Datasets
    merged_df = pd.merge(poverty_df_world , MMR_df_world, on=["Country", "Year"] )

    # X and Y values
    X = merged_df[['PR']] * 100  # Double brackets makes it 2D
    Y = merged_df['MMR']

    # Fit for linear regression
    model = LinearRegression()
    model.fit(X, Y)

    # Prediction for Y
    Y_pred = model.predict(X)


    # Plotting graph
    plt.figure(figsize=(8, 6)) # Creates a new figures for each plot
    plt.scatter(X, Y, label='Data Points') 
    plt.plot(X, Y_pred, linewidth=2,color = 'red',label='Regression Line') 
    plt.title('Linear Regression for Poverty Rate and Maternal Mortality Rate')
    plt.xlabel('Poverty Rate %')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    plt.legend()
    plt.grid(True)
    plt.show()

    r_value = r2_score(Y, Y_pred)

    return r_value


# ---------------------------------------------------------------------
# Hypothesis 2
# Countries which have similar levels of poverty (low, high, medium) will have differing MMR depending on education level

# ---------------------------------------------------------------------

#Next steps: Calc correlation for each group

# Make differing scatter plots two ways: one with countries put in diferent ifnancial categories by Ml
# Other way: using dataset of countries in different known classificaiton by WBG

def plot_bubble_plot():
    
    # Makes values for Year consistent types and coerce sets invalid parsing to NaN
    poverty_df["Year"] = pd.to_numeric(poverty_df["Year"], errors="coerce")
    education_df["Year"] = pd.to_numeric(education_df["Year"], errors="coerce")
    MMR_df["Year"] = pd.to_numeric(MMR_df["Year"], errors="coerce")

    # Filter    
    poverty_df_world = poverty_df[poverty_df["Country"] == "World"]
    education_df_world = education_df[education_df["Country"] == "World"]
    MMR_df_world = MMR_df[MMR_df["Country"] == "World"]

    # Merge Datasets
    merged_df = poverty_df_world.merge(MMR_df_world, on=["Country", "Year"])
    merged_df = merged_df.merge(education_df_world, on=["Country", "Year"])

    # X and Y values
    X_1D = merged_df['PR']
    X = merged_df[['PR']] * 100  # Double brackets makes it 2D
    Y = merged_df['MMR']

    # Bubbles
    bubbles = merged_df['PCR']

    # Fit for linear regression
    model = LinearRegression()
    model.fit(X, Y)

    # Prediction for Y
    Y_pred = model.predict(X)

    # Plotting graph
    plt.figure(figsize=(8, 6)) # Creates a new figures for each plot
    scatter_plt= plt.scatter(X, Y, s=60, alpha= 0.5, c=bubbles, cmap='winter_r', label='Data Points') 
    plt.plot(X, Y_pred, linewidth=2,color = 'red',label='Regression Line') 

    # A color bar to show GDP scaled
    cbar = plt.colorbar(scatter_plt)
    cbar.set_label('Primary completion rate, total (%)', fontsize=10)
    plt.title('Linear Regression for Poverty Rate and Maternal Mortality Rate')
    plt.xlabel('Poverty Rate %')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_income_group_scatter(income_group):

    education_df["Year"] = pd.to_numeric(education_df["Year"], errors="coerce")
    MMR_df_income["Year"] = pd.to_numeric(MMR_df_income["Year"], errors="coerce")

    # Plotting both dfs told only include 'World' entity
    education_df_income = education_df[education_df["Country"] == income_group]
    mmr_income = MMR_df_income[MMR_df_income["Income group"] ==income_group]

    # For merging datasets
    education_df_income = education_df_income.rename(columns={
        'Country': 'Income group' })
    
    # Merge Datasets
    merged_df = pd.merge(education_df_income , mmr_income, on=["Income group", "Year"] )

    # LinearRegression doesnt accept (X) input with NaN
    merged_df = merged_df.dropna(subset=['PCR', 'Mean_MMR'])

    # X and Y values
    X = merged_df[['PCR']] # Double brackets makes it 2D
    Y = merged_df['Mean_MMR']

    # Fit for linear regression
    model = LinearRegression()
    model.fit(X, Y)

    # Prediction for Y
    Y_pred = model.predict(X)

    # Plotting graph
    plt.figure(figsize=(8, 6)) # Creates a new figures for each plot
    plt.scatter(X, Y, label='Data Points') 
    plt.plot(X, Y_pred, linewidth=2,color = 'red',label='Regression Line') 
    plt.title(f'{income_group}')
    plt.xlabel('Primary completion rate %')
    plt.ylabel('Mean Maternal Mortality Rate (Deaths per 100,000)')
    plt.legend()
    plt.grid(True)
    plt.show()

    r_value = r2_score(Y, Y_pred)

    return f"{income_group}'s correlation is : {r_value}"

# Call all df functions

poverty_df = read_poverty_data()
education_df = read_education_data()
MMR_df = read_MMR_data()
MMR_df_income = read_MMR_income_data()


# Funcion to create boxplots of incoem groups
def box_plots():

    MMR_df_income.boxplot(
        column="Mean_MMR",
        by="Income group"
    )
    # Plotting the Graph
    plt.suptitle(" ") # Gets rid of automatic graph title
    plt.title("Distribution of Mean MMR by Income Group")
    plt.xlabel("Income Group")
    plt.ylabel('Mean Maternal Mortality Rate (Deaths per 100,000)')
    plt.grid(axis="y")
    plt.show()


if __name__ == "__main__":
   
    # Plots scatter plot for poverty and MMR globally 
    scatter_plot= plot_scatter_poverty_MMR()

    # Plots bubble plot for global data
    bubble_plot= plot_bubble_plot()

    # Plots Income groups graphs
    income_groups = MMR_df_income["Income group"].unique()
    for income_group in income_groups:
        income_groups = plot_income_group_scatter(income_group)
        print(income_groups)

    box_plot = box_plots()
        