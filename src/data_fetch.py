import pandas as pd
import requests



import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

url = "https://api.themoviedb.org/3/movie/{}?api_key={}&append_to_response=credits" 

movie_ids = [0, 299534, 19995, 140607, 299536, 597, 135397,
 420818, 24428, 168259, 99861, 284054, 12445,
 181808, 330457, 351286, 109445, 321612, 260513]  

data = []

for movie_id in movie_ids:
    response = requests.get(url.format(movie_id, api_key))
    if response.status_code == 200:
        data.append(response.json())

df = pd.DataFrame(data)
print(df.head())

print(df["credits"])



df.to_csv("output.csv", index=False)
