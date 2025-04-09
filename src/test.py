import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_cleaning import drop_columns
import pandas as pd

columns_to_drop = ['adult', 'imdb_id', 'original_title', 'video', 'homepage']

df = df = pd.read_csv("../output.csv")
drop_columns(df, columns_to_drop)

# df = clean_json_columns(df)