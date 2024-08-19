# Cody Giles - Student ID: 010506641
# C964 Capstone - Movie Audience Rating Predictor aka The MARP

# import csv
import pandas as pd

# Bring in the dataset CSV
dfCSV = pd.read_csv('imdb_movies - test.csv')

dfCSV = pd.DataFrame(
    {'test': []}
)

print(dfCSV)