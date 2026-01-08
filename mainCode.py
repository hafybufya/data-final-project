# ---------------------------------------------------------------------
# IMPORTED FUNCTIONS USED IN PROGRAM
# ---------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Figures directoy name
figures_directory = 'figures'
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

#=== POVERTY WORlD DF ===
def read_poverty_data(start_year, end_year):

    """

    Loads the csv dataset defined in 'poverty_csv_region'
    
    Paramters
    ---------
    start_year : int, min year of dataset to be returned
    end_year : int, end year of dataset to be returned

    Returns
    -------

    pandas Dataframe -> converts csv to df containing poverty data

    """
        
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


#=== EDUCATION DF ===
def read_education_data(start_year, end_year):

    """

    Loads the csv dataset defined in 'mmr_csv'
    
    Paramters
    ---------
    start_year : int, min year of dataset to be returned
    end_year : int, end year of dataset to be returned

    Returns
    -------

    pandas Dataframe -> converts csv to df containing education data

    """
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

    education_df["Year"] = pd.to_numeric(education_df["Year"], errors="coerce")
    education_df = education_df[(education_df["Year"] >= start_year) & (education_df["Year"] <= end_year)]
    return education_df


#=== MMR DF ===
def read_mmr_data(start_year, end_year):
    """

    Loads the mmr dataset defined in 'mmr_csv'
    
    Paramters
    ---------
    start_year : int, min year of dataset to be returned
    end_year : int, end year of dataset to be returned

    Returns
    -------

    pandas Dataframe -> converts csv to df containing MMR data

    """
    df = pd.read_csv(mmr_csv)
    mmr_df =  df[['DIM_TIME', 'GEO_NAME_SHORT',  'RATE_PER_100000_N']].copy()

    mmr_df = mmr_df.rename(columns={
        'DIM_TIME': 'Year',
        'GEO_NAME_SHORT': 'Country',
        # MMR = Maternal deaths per 100,000 births
        'RATE_PER_100000_N': 'MMR'
    })
    mmr_df["Year"] = pd.to_numeric(mmr_df["Year"], errors="coerce")
    mmr_df = mmr_df[(mmr_df["Year"] >= start_year) & (mmr_df["Year"] <= end_year)]
    return  mmr_df


#=== MMR INCOME DF ===
def read_mmr_income_data():

    """

    Loads the mmr dataset defined in income_classification_csv to be used to
    group Mean MMR by income groups and Year.
    

    Returns
    -------
    pandas Dataframe -> merged dataframe containing MMR data based on income group

    """

    #Columns: Year, Income group, Mean MMR 
    df = pd.read_csv(income_classification_csv)
    income_df =  df[['Economy', 'Income group']].copy()
    income_df = income_df.rename(columns={
        # Same name for merging with MMR df
        'Economy': 'Country'})
    
    # Merge Datasets
    merged_df = pd.merge(mmr_df , income_df, on=["Country"] )
    
    result = (
        merged_df
        # as_index = False ensures grouping stays as normal columsn
        .groupby(["Year", "Income group"], as_index=False).agg(Mean_MMR=("MMR", "mean"))
    )

    return result


#=== WORLD FILTERS ===
def world_filters(column, world):
    """

    Filters each df by world entity 
        
    Paramters
    ---------
    country : str, column in df where filtering is done on
    world : str, value filtering each df by

    Returns
    -------
    poverty_df_world   : pandas Dataframe which contains World Poverty data
    education_df_world : pandas Dataframe which contains World Education data
    mmr_df_world       : pandas Dataframe which contains World MMR data

    """

    poverty_df_world = poverty_df[poverty_df[column] == world]
    education_df_world = education_df[education_df[column] == world]
    mmr_df_world = mmr_df[mmr_df[column] == world]

    return poverty_df_world, education_df_world, mmr_df_world


# ---------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------

#=== SCATTER PLOT OF POVERTY AND MMR ===
def plot_scatter_poverty_mmr():
    """

    Creates a scatter plot for global poverty against global MMR

    Returns
    -------
    X_1D            : int, returns 1 dimensional X values
    X               : int, returns 2 dimensional X values
    merged_df       : pandasDataframe, returns merged_df for world poverty and mmr values
    slope           : int, returns slope of line in plot
    r_value.        : int, returns pearson correlation value
    r_squared_value : int, returns r squared value

    """

    # Merge Datasets
    merged_df = pd.merge(poverty_df_world , mmr_df_world, on=["Country", "Year"])

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
    plt.plot(X, Y_pred, linewidth=2,color = 'red', label=f'Regression line \n y={slope:.2f}x + {intercept:.2f}') 
    plt.title('Global Poverty Rate vs Global Maternal Mortality Rate')
    plt.xlabel('Poverty Rate %')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    txt="Figure 1: Scatter Plot showing the relationship between poverty rate and maternal mortality ratio on a Global Scale. Each point represents the global maternal mortality for a given year. The line represents the linear regression line."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Reserve space at the bottom for caption
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{figures_directory}/scatter_global_poverty_vs_mmr.png")
    plt.show()

    r_squared_value = r2_score(Y, Y_pred)

    # Returning slope for statement for every 1% increase in poverty, mmr increases by slope amount
    return X_1D, X, merged_df, slope, r_value, r_squared_value

#=== GlOBAL BUBBLE PLOT ===
def plot_bubble_plot():

    """
    
    Creates a bubbleplot for poverty, education rate and MMRR

    """
    # Merge Datasets
    merged_df = poverty_df_world.merge(mmr_df_world, on=["Country", "Year"])
    merged_df = merged_df.merge(education_df_world, on=["Country", "Year"])

    # X and Y values
    X_1D = merged_df['PR'] *100
    X = merged_df[['PR']] * 100  # Double brackets makes it 2D
    Y = merged_df['MMR']

    # Bubbles
    bubbles = merged_df['PCR']

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(X_1D, Y)

    # Fit for linear regression
    model = LinearRegression()
    model.fit(X, Y)

    # Prediction for Y
    Y_pred = model.predict(X)

    # Plotting graph
    plt.figure(figsize=(8, 6)) # Creates a new figures for each plot
    scatter_plt= plt.scatter(X, Y,  alpha= 0.5, s=65, c=bubbles, cmap='winter_r', label='Data Points') 
    plt.plot(X, Y_pred, linewidth=2,color = 'red', label=f'Regression line \n y={slope:.2f}x + {intercept:.2f}') 

    # A color bar to show GDP scaled
    cbar = plt.colorbar(scatter_plt)
    cbar.set_label('Primary completion rate, total (%)', fontsize=10)
    plt.title('Global Poverty Rate vs Global Maternal Mortality Rate')
    plt.xlabel('Poverty Rate %')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    plt.legend()
    txt="Bubble Plot showing the relationship between poverty rate, education completion and maternal mortality ratio on a Global Scale. Each point represents the global maternal mortality for a given year. The line represents the linear regression line. The cbar represents education rate."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.grid(True)
    plt.tight_layout()
    # Reserve space at the bottom for caption
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(f"{figures_directory}/bubble_global_poverty_vs_mmr.png")
    plt.show()


#=== BOXPLOT FOR 2023 VALUES ===
def box_plots():
    """

    Creates a box plot for MMR, divided by different income groups for 
    the most recent year of data: 2023

    Returns
    -------
    merged_df : pandasDataframe, returns merged_df for income group and mmr values

    """

    mmr_2023 = mmr_df[mmr_df["Year"] == 2023]
    income_df = pd.read_csv(income_classification_csv)
    income_df = income_df.rename(columns={'Economy': 'Country'})

    merged_df = mmr_2023.merge(income_df, on=["Country"])
    fig, ax = plt.subplots(figsize=(8, 7))
    merged_df.boxplot( column="MMR", by="Income group" , ax=ax)

 
    # Gets most extreme mmr value
    extreme_row = merged_df.loc[merged_df["MMR"].idxmax()]

    # Gets y axis value
    extreme_mmr = extreme_row["MMR"]
     
    # Gets country name
    extreme_country = extreme_row["Country"]

    # Gets income group of extreme value
    extreme_income = extreme_row["Income group"]

    # Find x-position of the income group box
    income_groups = merged_df["Income group"].sort_values().unique()
    x_pos = list(income_groups).index(extreme_income) 

    # Plot and label the extreme point
    ax.scatter(x_pos, extreme_mmr, color="red", s=80, zorder=5)

    ax.annotate(
        f"{extreme_country}",
        xy=(x_pos, extreme_mmr),
        xytext=(x_pos + 0.3, extreme_mmr + 20),
        arrowprops=dict(arrowstyle="->", color="blue"),
        fontsize=9,
        color="blue"
    )

    # Plotting the Graph
    plt.suptitle(" ") # Gets rid of automatic graph title
    plt.title("Distribution of Maternal Mortality Rate by Income Group")
    plt.xlabel("Income Group")
    plt.ylabel('Mean Maternal Mortality Rate (Deaths per 100,000)')
    # Add horizontal line at y = 70 (70 mmr)
    plt.axhline(y=70, linewidth=1, color= 'red')
    plt.text(
    0.02, 70, 'mmr = 70',
    verticalalignment='bottom', bbox ={'facecolor':'grey', 'alpha':0.2})
    txt="Figure 3: Boxplots showing the distribution of maternal mortality ratio in different World Bank income groups in 2023 (most recent available year of data). Each data point represent a country."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.grid(axis="y")
    plt.tight_layout()
    # Reserve space at the bottom for caption
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(f"{figures_directory}/boxplot_income_education_vs_mmr.png")
    plt.show()

    return merged_df


#=== INCOME GROUP SCATTER PLOT ===
def plot_income_group_scatter(income_group):
    """

    Creates a scatter plot for education rate and MMR for each income group

    Returns
    -------
    merged_df       : pandasDataframe, returns merged_df for povery,education and MMR
    r_value.        : int, returns pearson correlation value
    r_squared_value : int, returns r squared value

    """
    # Plotting both dfs told only include 'certain income group' entity
    education_df_income = education_df[education_df["Country"] == income_group]
    mmr_income = mmr_df_income[mmr_df_income["Income group"] ==income_group]

    # For merging datasets
    education_df_income = education_df_income.rename(columns={
        'Country': 'Income group' })
    
    # Merge Datasets
    merged_df = pd.merge(education_df_income , mmr_income, on=["Income group", "Year"] )

    # LinearRegression doesnt accept (X) input with NaN
    merged_df = merged_df.dropna(subset=['PCR', 'Mean_MMR'])

    # X and Y values
    X_1D = merged_df['PCR']
    X = merged_df[['PCR']] # Double brackets makes it 2D
    Y = merged_df['Mean_MMR']

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
    plt.plot(X, Y_pred, linewidth=2,color = 'red', label=f'Regression line \n y={slope:.2f}x + {intercept:.2f}') 
    plt.title(f'{income_group}: Primary Completion Rate vs Mean Maternal Mortality Rate')
    plt.xlabel('Primary completion rate %')
    plt.ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    txt=f"Scatter Plot showing the relationship between education rate and mean maternal mortality ratio for {income_group}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Reserve space at the bottom for caption
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(f"{figures_directory}/timeseries_{income_group}_mmr.png")
    plt.show()

    r_squared_value = r2_score(Y, Y_pred)

    return merged_df, r_value, r_squared_value


#=== HIGH LOW INCOME SCATTER PLOTS ====
def plot_high_low_income_scatter():
    """

    Places both high income and low income scatter on the same figure 
    allowing for comparison of education rate and MMR.
    
    """
    fig, (ax_high, ax_low) = plt.subplots(1, 2, figsize=(15, 6))

    # high income
    plot_single_income("High income", ax_high)
 
    # low income
    plot_single_income("Low income", ax_low)

    ax_high.legend()
    ax_low.legend() 
    plt.tight_layout()
    
    txt="Figure 2: Scatter Plot showing the relationship between primary completion rate and maternal mortality ratio in High and Low Income Groups. Each point represents the Mean maternal mortality for each given year. The line represents the linear regression line. "
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.tight_layout()
    # Reserve space at the bottom for caption
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(f"{figures_directory}/scatter_high_low_pov_mmr.png")
    plt.show()


#=== INVIDIUALLY PLOT INCOME GROUP ===
def plot_single_income(income_group, ax):
    """

    Helper function for plot_high_low_income_scatter()
    Plots a scatter plot showing the relationship between 
    education rate and MMR for a specified income group.

    Paramaters
    -------
    income_group : str, filters 'Country' column for income_group
    ax           : matplotlib.axes.Axes, Matplotlib axis on which the plot is drawn.

    """

    education_df_income = education_df[education_df["Country"] == income_group]
    mmr_income = mmr_df_income[mmr_df_income["Income group"] == income_group]

    education_df_income = education_df_income.rename(
        columns={'Country': 'Income group'}
    )

    merged_df = pd.merge(
        education_df_income,
        mmr_income,
        on=["Income group", "Year"]
    )

    merged_df = merged_df.dropna(subset=['PCR', 'Mean_MMR'])
 
    X_1D= merged_df['PCR']
    X = merged_df[['PCR']]
    Y = merged_df['Mean_MMR']


    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(X_1D, Y)


    ax.scatter(X, Y, label='Data Points')
    ax.plot(X, Y_pred, color='red', linewidth=2, label=f'Regression line \n y={slope:.2f}x + {intercept:.2f}')

    ax.set_title(f'{income_group}: Primary Completion Rate vs Mean Maternal Mortality Rate')
    ax.set_xlabel("Primary completion rate %")
    ax.set_ylabel("Maternal Mortality Rate")

    ax.grid(True)


#=== WORLD NIGERIA SCATTER PLOT ===
def plot_world_nigeria_timeseries():
    """

    Places both World and Nigeria scatter on the same figure 
    allowing for comparison between two regions

    Returns
    -------
    world_change.  : float, % change in MMR from start to end year globally
    nigeria_change : float, % change in MMR from start to end year in Nigeria

    """

    fig, (ax_world, ax_nigeria) = plt.subplots(1, 2, figsize=(15, 6))

    # high income
    world_change = plot_timeseries_plot("World", ax_world)

    # low income
    nigeria_change = plot_timeseries_plot("Nigeria", ax_nigeria)

    ax_world.legend()
    ax_nigeria.legend()
    
    plt.tight_layout()
    # Reserve space at the bottom for caption
    plt.subplots_adjust(bottom=0.10)

    txt="Timeseries showing maternal mortality ratio overtime globally and in Nigeria."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9, bbox ={'facecolor':'grey', 'alpha':0.2})
    plt.savefig(f"{figures_directory}/timeseries_world_nigeria_mmr.png")
    plt.show()

    return world_change, nigeria_change


#=== INVIDIUALL PLOT REGION ===
def plot_timeseries_plot(country, ax):
    """

    Helper function for plot_world_nigeria_timeseries()
    Plots a timeseries plot showing the relationship between 
    MMR for a specified region

    Paramaters
    -------
    country : str, filters 'Country' column for country i.e. World and Nigera
    ax           : matplotlib.axes.Axes, Matplotlib axis on which the plot is drawn.

    Returns
    -------
    result  : float, % change in MMR from start to end year based on passed in 'country'
    """

    mmr_df_filtered = mmr_df[mmr_df["Country"] == country].sort_values("Year")   
    mmr_df_filtered = mmr_df_filtered.dropna(subset=['Year', 'MMR'])

    ax.plot(mmr_df_filtered['Year'], mmr_df_filtered['MMR'], label= country
    )

    ax.set_title(f"Time Series on Global Maternal Moratlity Rate {start_year}-{max_year}")
    ax.set_xlabel('Year')
    ax.set_ylabel('Maternal Mortality Rate (Deaths per 100,000)')
    ax.grid(True)

    # Percentage change from start to end year
    mmr_min_value = mmr_df_filtered.loc[mmr_df_filtered["Year"] == start_year, "MMR"].iloc[0]
    mmr_max_value = mmr_df_filtered.loc[mmr_df_filtered["Year"] == max_year, "MMR"].iloc[0]

    result= ((mmr_max_value - mmr_min_value)/ mmr_min_value)* 100

    # float() to convert from np.float64 to a normal float
    return float(result)



# ---------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------

#=== GLOBAL MMR CALCULATIONS ===
def get_global_mmr_values():
    """
    Calculates global MMR values to be included in the report

    Returns
    -------
    mmr_2023_value  : float, returns global MMR value in 2023
    percentage_lmic : float, returns the % of maternal deaths that occur in LMICs

    """
    mmr_2023_value = mmr_df_world.loc[mmr_df_world["Year"] == 2023, "MMR"].iloc[0]
    
    mmr_df_income_2023 = mmr_df_income[mmr_df_income["Year"] == 2023]

    lmic_mmr = mmr_df_income_2023[
        mmr_df_income_2023["Income group"].isin(
            ["Low income", "Lower middle income"]
        )
    ]["Mean_MMR"].sum()

    total_mmr = mmr_df_income_2023["Mean_MMR"].sum()

    percentage_lmic = (lmic_mmr / total_mmr) * 100
    return mmr_2023_value, percentage_lmic


#=== PERCENTAGE IN POVERTY ===
def percentage_pop_poverty():
    """
    Calculates percentage of the population living in poverty in 2023

    Returns
    -------
    pov_2023_value  : float, returns %

    """
    pov_2023_value = poverty_df_world.loc[poverty_df_world["Year"] == 2023, "PR"].iloc[0]
    return pov_2023_value * 100


# Prints each country with highest MMR each year
def highest_mmr_by_year():
    """
    Prints which country has the highest MMR for the period of data 

    Returns
    -------
    highest  : df, containg Year, Country and MMR

    """
    highest = (
        mmr_df.loc[mmr_df.groupby("Year")["MMR"].idxmax()]
        .sort_values("Year")
        [["Year", "Country", "MMR"]]
    )
    return highest

 
# Call all df functions
start_year = 1985
max_year = 2023

poverty_df = read_poverty_data(start_year, max_year)
education_df = read_education_data(start_year, max_year)
mmr_df = read_mmr_data(start_year, max_year)
mmr_df_income = read_mmr_income_data()


poverty_df_world, education_df_world, mmr_df_world = world_filters("Country", "World")


if __name__ == "__main__":
    
    # pov_2023_column = percentage_pop_poverty()
    # print (pov_2023_column)

    plot_world_nigeria_mmr = plot_world_nigeria_timeseries()
    print(plot_world_nigeria_mmr)


    mmr_2023_value, percentage_lmic = get_global_mmr_values()
    print(f"mmr 2023 value is {mmr_2023_value}") 
    print(f"Percentage of mmr in LMICs {percentage_lmic}")
    # Plots scatter plot for poverty and mmr globally 

    X_1D, X, merged_df, slope , correlation_coefficient , r_squared_value = plot_scatter_poverty_mmr()
    print(f"Slope for Global Poverty vs Global mmr: {slope}")
    print(f"Correlation Coefficient for Global Poverty vs Global mmr: {correlation_coefficient}")
    print(f"R squared value for Global Poverty vs Global mmr: {r_squared_value}")

    # Plots bubble plot for global data
    bubble_plot= plot_bubble_plot()

    # Plots Income groups graphs
    income_groups = mmr_df_income["Income group"].unique()
    for income_group in income_groups:
        merged_df, r_value, r_squared_value = plot_income_group_scatter(income_group)
        print(f"{income_group}'s correlation is : {r_value}")

    high_low_scatter = plot_high_low_income_scatter()


    box_plot = box_plots()
    print(box_plot)

    highest = highest_mmr_by_year()