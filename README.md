# Data Final Project

Final Project for Analysis, Software and Career Practice Year 2 Module.

## 📊 Datasets

The datasets used were obtained from The World Bank Group (WBG) and the World Health Organisation (WHO):

1. WBG - [Poverty and Inequalities by Region](https://pip.worldbank.org/nowcasts) 
2. WBG - [Primary completion Rate, total (% of relevant age group)](https://data.worldbank.org/indicator/SE.PRM.CMPT.ZS)
3. WHO - [Maternal mortality Ratio (per 100 000 live births)](https://data.who.int/indicators/i/C071DCB/AC597B1)
4. WBG - [Income Group Classification](https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups)


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

## 🛠️ Installation

Python 3.10 or newer to run python files

Python modules required:

* pandas – reading and handling CSV files.
* matplotlib – plotting graphs.
* os - checking if files exist.
* scipy - 
* sklearn - 

You can install required packages with:

```
pip install pandas matplotlib os scipy sklearn

```

