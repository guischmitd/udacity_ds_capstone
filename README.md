# Udacity + Arvato Customer Segmentation Challenge
This repo is one of the deliverables of the Capstone Project in the Data Scientist Nanodegree program by Udacity.

## Project Context
The Arvato Financial Solutions Customer Segmentation Competition is a challenge created in partnership with Udacity designed to test many of the skills required of a Data Science professional, from data exploration, analysis and transformation, all the way through modeling, optimizing and generating predictions. The data kindly provided by Arvato Bertelsmann contain a large number of features and samples of individuals divided in two groups: Customers (of a mailout company for which Arvato is consulting) and general population. Features range in scope from individual characteristics, such as age and gender, to household and community, such as unemployment rate and number of cars. All samples relate to people in Germany.
The task at hand is divided in three parts, each with its own ipython notebook:
1. Data exploration (`1_exploration_and_cleaning.ipynb`), in which we're supposed to analyze, transform and clean the raw data.
2. Unsupervised learning (`2_customer_segmentation.ipynb`), consisting of determining trends in the demographics of customers to better understand the traits that make an individual more prone to becoming a customer.
3. Supervised learning (`3_suvervised_learning_model.ipynb`), which asks for a model that correctly predicts the probability of individuals to become customers.

## Quickstart
It is recommended that you create a specific environment for this project
```bash
# On Linux
virtualenv .venv -p python3
source .venv/bin/activate

# On Windows Powershell
python -m virtualenv .venv
./.venv/Scripts/Activate.ps1

# or on Windows Git Bash
python -m virtualenv .venv
source ./.venv/Scripts/activate
```
Install dependencies, then open each jupyter notebook in your preferred way (vscode extension or jupyter kernel running locally/on the cloud)
```bash
pip install -r requirements.txt
jupyter notebook
```

*Note*: Unfortunately the provided data is not to be redistributed by students, so you'll have to be enrolled in the same project or acess the material via kaggle. It is possible to follow my work directly on github by opening the jupyter notebooks used in each step of development.

## Running the preprocessing script
The `preprocess_data.py` script can be called from the command line, like so:
```bash
python preprocess_data.py -i {input_csv_path} -o {output_parquet_path} [optional]
```
Or it can be imported and ran as a function in a python script/ipython notebook:
```python
from preprocess_data import preprocess

path = 'path/to_my/dataset.csv'
raw_data = pd.read_csv(path)
preprocessed_data = preprocess(raw_data)
```