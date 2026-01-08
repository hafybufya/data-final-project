# Data Final Project

This is my Final Project for the year 2 module: Analysis, Software and Career Practice. This project investigates the relationship between poverty, education, and maternal mortality rates at both global, income-group and country level.

### 🧪 Hypothesis 

A combination of poverty and limited access to education contribute to a higher global maternal mortality ratio.
To examine this relationship in depth, this hypothesis is broken up into two sub-hypotheses:
1.    There is a relationship between poverty and maternal mortality ratio.
2.    Within income groups, primary education completion is associated with a lowered maternal mortality ratio

## ✨ Key Features
* Cleaning and processing of multiple authentic datasets
* Global level and income group level and country level analysis
* Scatter plots, timeseries plots, bubble plots, and box plots
* Linear regression, correlation coefficients, and R² values
* Vigorous unit testing of functions

## 💻 AI Assistance
Some of the code and debugging in this project were developed with the assistance of chatGPT.
AI was used to:
* Calculate and label outlier on the boxplot
* Assisting with the code to create subplots 
* Provide example unittests to code

All analysis and the accomponying written report were done by the author (myself).


## 📁 Project Structure

```
├── mainCode.py
├── unitTesting.py
├── requirements.txt
├── data/
│   ├── poverty_region.csv
│   ├── education.csv
│   ├── MMR.csv
│   └── income_group_classification.csv
├── figures/
│   ├── boxplot_income_education_vs_mmr
│   ├── bubble_global_poverty_vs_mmr
│   ├── global_poverty_vs_mmr
│   ├── scatter_high_low_pov_mmr
│   ├── timeseries_High income_mmr
│   ├── timeseries_Low income_mmr
│   ├── timeseries_Upper middle income_mmr
│   ├── timeseries_Lower middle income_mmr
│   └── timeseries_world_nigeria_mmr
├── LICENSE
├── README.md
└── .circleci/
    └── config.yml

```

## 📊 Datasets

The datasets used were obtained from The World Bank Group (WBG) and the World Health Organisation (WHO):

1. WBG - [Poverty and Inequalities by Region](https://pip.worldbank.org/nowcasts) 
2. WBG - [Primary completion Rate, total (% of relevant age group)](https://data.worldbank.org/indicator/SE.PRM.CMPT.ZS)
3. WHO - [Maternal mortality Ratio (per 100 000 live births)](https://data.who.int/indicators/i/C071DCB/AC597B1)
4. WBG - [Income Group Classification](https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups)

All datasets are loaded from the ```data/``` directory and processed using ```pandas```
Data used in this investigation was all publically available.

## 🎯 Conclusions

* There is a relationship between poverty, education and maternal mortality but the strength of the relationship differs substaintially between income groups.
* Education rates and maternal mortality rates have a strong negative relationship within low income and lower-middle income countries (r = -0.97) -> as education increases maternal mortality rate decrease
* In high income countries,other macrofactors maybe be at play where education rate is near universal 
* Future research should evaluate other macrofactors including quality of available healthcare. 


## 📝 Unit Tests

Unit tests are completed using Python’s builtin unittest framework. 
Tests include:
* Checking csv files exist
* Verifying dataframe columns and merged results
* Ensuring calculated correlations and R² values are valid
* Confirming numerical outputs for percentages and values


## 🛠️ Installation

Python 3.10 or newer to run python files

Python modules required:

* pandas – reading and handling CSV files.
* matplotlib – plotting graphs.
* os - checking if files exist.
* scipy - statistical calculations
* sklearn -  linear regression and modeling

You can install required packages with:

```
pip install pandas matplotlib os scipy sklearn

```

