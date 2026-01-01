# ---------------------------------------------------------------------
# IMPORTED FUNCTIONS USED IN PROGRAM
# ---------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# CSVs used in programs defined
csv_folder = 'data'
poverty_csv_region = f'{csv_folder}/poverty_region.csv'
education_csv = f'{csv_folder}/education.csv'
mmr_csv= f'{csv_folder}/maternal_mortality_rate.csv'
income_classification_csv = f'{csv_folder}/income_group_classification.csv'

# ---------------------------------------------------------------------
# DATA CLEANING SECTION
# -> Including only necessary columns in dataFrames
# -> Renaming columns for esaier analysis
# ---------------------------------------------------------------------

# POVERTY WORlD DF 

def read_poverty_data(start_year, end_year):

    poverty_df = pd.read_csv(poverty_csv_region)
    poverty_df =  poverty_df[['region_name', 'reporting_year', 'headcount']].copy()

    poverty_df = poverty_df.rename(columns={
        'reporting_year': 'Year',
        'region_name': 'Country',
        # PR = Poverty Rate (%)
        'headcount': 'PR'
    })
    poverty_df["Year"] = pd.to_numeric(poverty_df["Year"], errors="coerce")
    poverty_df = poverty_df[(poverty_df["Year"] >= start_year) & (poverty_df["Year"] <= end_year)]
    return poverty_df

# EDUCATION DF
def read_education_data(start_year, end_year):

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
    education_df["Year"] = pd.to_numeric(education_df["Year"], errors="coerce")
    education_df = education_df[(education_df["Year"] >= start_year) & (education_df["Year"] <= end_year)]
    return education_df

# MMR DF
def read_MMR_data(start_year, end_year):

    df = pd.read_csv(mmr_csv)
    MMR_df =  df[['DIM_TIME', 'GEO_NAME_SHORT',  'RATE_PER_100000_N']].copy()

    MMR_df = MMR_df.rename(columns={
        'DIM_TIME': 'Year',
        'GEO_NAME_SHORT': 'Country',
        # MMR = Maternal deaths per 100,000 births
        'RATE_PER_100000_N': 'MMR'
    })
    MMR_df["Year"] = pd.to_numeric(MMR_df["Year"], errors="coerce")
    MMR_df = MMR_df[(MMR_df["Year"] >= start_year) & (MMR_df["Year"] <= end_year)]
    return  MMR_df

# MMR INCOME DF
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
    #result.to_csv('filename.csv', index=False)

    return result


def world_filters():
    poverty_df_world = poverty_df[poverty_df["Country"] == "World"]
    education_df_world = education_df[education_df["Country"] == "World"]
    MMR_df_world = MMR_df[MMR_df["Country"] == "World"]

    return poverty_df_world, education_df_world, MMR_df_world

# ---------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------

#Future Improvement: Adding a heatmap for easier visualisation

def plot_scatter_poverty_MMR():

    # Merge Datasets
    merged_df = pd.merge(poverty_df_world , MMR_df_world, on=["Country", "Year"] )

    # X and Y values
    X_1D = merged_df['PR'] * 100 # For lineregress
    X = merged_df[['PR']] * 100  # Double brackets makes it 2D
    Y = merged_df['MMR']

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(X_1D, Y)

    # Fit for linear regression
    model = LinearRegression()
    model.fit(X, Y)


    # Prediction for Y
    Y_pred = model.predict(X)

    # Plotting graph
    plt.figure(figsize=(8, 6)) # Creates a new figures for each plot
    plt.scatter(X, Y, label='Data Points') 
    plt.plot(X, Y_pred, linewidth=2,color = 'red',label='Regression Line') 
    plt.title('Global Poverty Rate vs Global Maternal Mortality Rate')
    plt.xlabel('Poverty Rate %')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    txt="Scatter Plot showing the relationship between Poverty Rate and Maternal Mortality Rate on a Global Scale."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.grid(True)
    plt.show()

    r_squared_value = r2_score(Y, Y_pred)

    # Returning slope for statement for every 1% increase in poverty, MMR increases by slope amount
    return X_1D, X, merged_df, slope, r_value, r_squared_value


# Funcion to create boxplots of income groups 
# Countries organised in the most recent year -> 2023 
def box_plots():
  
    mmr_2023 = MMR_df[MMR_df["Year"] == 2023]
    income_df = pd.read_csv(income_classification_csv)
    income_df = income_df.rename(columns={'Economy': 'Country'})

    merged_df = mmr_2023.merge(income_df, on=["Country"])

    #merged_df.to_csv('filename.csv', index=False)

    merged_df.boxplot(
        column="MMR",
        by="Income group"
    )
    # Plotting the Graph
    plt.suptitle(" ") # Gets rid of automatic graph title
    plt.title("Distribution of Maternal Mortality Rate by Income Group ")
    plt.xlabel("Income Group")
    plt.ylabel('Mean Maternal Mortality Rate (Deaths per 100,000)')
    # Add horizontal line at y = 70 (70 MMR)
    plt.axhline(y=70, linewidth=1, color= 'red')
    plt.text(
    0.02, 70, 'MMR = 70',
    verticalalignment='bottom', bbox ={'facecolor':'grey', 'alpha':0.2})

    txt="Box Plots showing the distribution of MMR in different income groups. Each point represent a country."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.grid(axis="y")
    plt.show()

    return merged_df


def plot_time_income_mmr_series():
    mmr_df = MMR_df_income.copy()
    mmr_df["Year"] = pd.to_numeric(mmr_df["Year"], errors="coerce")
    mmr_df = mmr_df.sort_values("Year") # Organises Year to prevent spaghetti time series

    df = mmr_df.dropna(subset=["Year", "Mean_MMR", "Income group"])

    income_groups = mmr_df["Income group"].unique()

    for income_group in income_groups:
        group_df = df[df["Income group"] == income_group]
        group_df = group_df.sort_values("Year")

        plt.plot(
            group_df["Year"],
            group_df["Mean_MMR"],
            label=income_group
        )

    x =  df["Year"]
    y =  df["Mean_MMR"] 
    plt.title(" Time Series on Global Maternal Moratlity Rate (1985-2023) ")  
    plt.xlabel('Year')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    plt.legend()
    plt.show()


def plot_bubble_plot():

    # Merge Datasets
    merged_df = poverty_df_world.merge(MMR_df_world, on=["Country", "Year"])
    merged_df = merged_df.merge(education_df_world, on=["Country", "Year"])

    # X and Y values
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
    scatter_plt= plt.scatter(X, Y,  alpha= 0.5, s=65, c=bubbles, cmap='winter_r', label='Data Points') 
    plt.plot(X, Y_pred, linewidth=2,color = 'red',label='Regression Line') 

    # A color bar to show GDP scaled
    cbar = plt.colorbar(scatter_plt)
    cbar.set_label('Primary completion rate, total (%)', fontsize=10)
    plt.title('Global Poverty Rate vs Global Maternal Mortality Rate')
    plt.xlabel('Poverty Rate %')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    txt="Bubble Plot showing the relationship between Poverty Rate, Maternal Mortality Rate and Education Completion on a Global Scale."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.grid(True)
    plt.show()

    return merged_df['PCR'].describe()

# Time series of global mmr in years
def plot_time_global_mmr_series():

    MMR_df_world = MMR_df[MMR_df["Country"] == "World"]
    # Redudant but not removing code until certain
    # MMR_df_world["Year"] = pd.to_numeric(MMR_df_world["Year"], errors="coerce")
    MMR_df_world = MMR_df_world.sort_values("Year") # Organises Year to prevent spaghetti time series

    x =  MMR_df_world["Year"]
    y =  MMR_df_world["MMR"] 
    plt.title(" Time Series on Global Maternal Moratlity Rate (1985-2023) ")  
    plt.xlabel('Year')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    plt.plot(x, y)
    plt.legend()
    plt.show()


# Plot based on income groups
def plot_income_group_scatter(income_group):

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
    plt.title(f'{income_group}: Primary Completion Rate vs Mean Maternal Mortality Rate')
    plt.xlabel('Primary completion rate %')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    txt=f"Scatter Plot showing the relationship between Education Rate and Mean Maternal Mortality for{income_group}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.grid(True)
    plt.show()

    r_value = r2_score(Y, Y_pred)

    return f"{income_group}'s correlation is : {r_value}"


def plot_high_low_income_scatter():

    fig, (ax_high, ax_low) = plt.subplots(1, 2, figsize=(15, 6))

    # high income
    plot_single_income("High income", ax_high)

    # low income
    plot_single_income("Low income", ax_low)

    plt.tight_layout()
    txt="Scatter Plot showing the relationship between Education Completion and Maternal Mortality Rate in High and Low Income Groups."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.show()


def plot_single_income(income_group, ax):

    education_df["Year"] = pd.to_numeric(education_df["Year"], errors="coerce")
    MMR_df_income["Year"] = pd.to_numeric(MMR_df_income["Year"], errors="coerce")

    education_df_income = education_df[education_df["Country"] == income_group]
    mmr_income = MMR_df_income[MMR_df_income["Income group"] == income_group]

    education_df_income = education_df_income.rename(
        columns={'Country': 'Income group'}
    )

    merged_df = pd.merge(
        education_df_income,
        mmr_income,
        on=["Income group", "Year"]
    )

    merged_df = merged_df.dropna(subset=['PCR', 'Mean_MMR'])

    X = merged_df[['PCR']]
    Y = merged_df['Mean_MMR']

    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    ax.scatter(X, Y)
    ax.plot(X, Y_pred, color='red', linewidth=2)

    ax.set_title(f"{income_group}")
    ax.set_xlabel("Primary completion rate %")
    ax.set_ylabel("Maternal Mortality Rate")

    ax.grid(True)

# ---------------------------------------------------------------------
# Estimating if UN will meet their goal of  7 maternal deaths per 
# 100,000 by 2030
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Calculations/Values for graphs
# ---------------------------------------------------------------------

# Get 2023 MMR value
# Get % of MMR that happen in low and low middle income countries
def get_global_MMR_values():

    mmr_2023_column = MMR_df_world[MMR_df_world["Year"] == 2023]
    
    MMR_df_income_2023 = MMR_df_income[MMR_df_income["Year"] == 2023]

    lmic_mmr = MMR_df_income_2023[
        MMR_df_income_2023["Income group"].isin(
            ["Low income", "Lower middle income"]
        )
    ]["Mean_MMR"].sum()

    total_mmr = MMR_df_income_2023["Mean_MMR"].sum()

    percentage_lmic = (lmic_mmr / total_mmr) * 100


    return mmr_2023_column, percentage_lmic


# Get percentage of population living on less than $3 a day 
def percentage_pop_poverty():
    pov_2023_column = poverty_df_world[poverty_df_world["Year"] == 2023]
    return pov_2023_column
    
    


# Call all df functions
start_year = 1985
max_year = 2023

poverty_df = read_poverty_data(start_year, max_year)
education_df = read_education_data(start_year, max_year)
MMR_df = read_MMR_data(start_year, max_year)
MMR_df_income = read_MMR_income_data()


poverty_df_world, education_df_world, MMR_df_world = world_filters()

if __name__ == "__main__":
    
    # pov_2023_column = percentage_pop_poverty()
    # print (pov_2023_column)

    # mmr_2023_value, percentage_lmic = get_global_MMR_values()
    # print(f"MMR 2023 column is {mmr_2023_value}") 
    # print(f"Percentage of MMR in LMICs {percentage_lmic}")
    # # Plots scatter plot for poverty and MMR globally 

    # X_1D, X, merged_df, slope , correlation_coefficient , r_squared_value = plot_scatter_poverty_MMR()
    # print(f"Slope for Global Poverty vs Global MMR: {slope}")
    # print(f"Correlation Coefficient for Global Poverty vs Global MMR: {correlation_coefficient}")
    # print(f"R squared value for Global Poverty vs Global MMR: {r_squared_value}")



    # # Plots bubble plot for global data
    # bubble_plot= plot_bubble_plot()

    # Plots Income groups graphs
    # income_groups = MMR_df_income["Income group"].unique()
    # for income_group in income_groups:
    #     income_groups = plot_income_group_scatter(income_group)
    #     print(income_groups)

    # high_low_scatter = plot_high_low_income_scatter()

    # plot = plot_time_income_mmr_series()
    # print(plot)

    box_plot = box_plots()
    print(box_plot)
