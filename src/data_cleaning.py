import pandas as pd
import ast

def drop_columns(df, columns_to_drop):
    df.drop(columns=columns_to_drop, inplace=True)
    return df



def clean_json_columns(df):
    """
    Cleans JSON-like columns in a movie dataset:
    - Extracts 'name' values from list of dicts.
    - Replaces original columns with cleaned strings.
    """
    # Define the columns to clean
    list_json_cols = [
        'genres', 'spoken_languages', 'production_countries', 'production_companies'
    ]

    dict_json_col = 'belongs_to_collection'  # this is a single dict

    # Parse stringified JSON safely
    def safe_parse(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith(("{", "[")) else x
        except:
            return None

    # Extract list of 'name' values joined by '|'
    def extract_names(val):
        if isinstance(val, list):
            return '|'.join([item.get('name', '') for item in val if 'name' in item])
        return None

    # Clean list-based columns
    for col in list_json_cols:
        df[col] = df[col].apply(safe_parse).apply(extract_names)

    # Clean 'belongs_to_collection' to keep only the collection name
    def extract_collection_name(val):
        if isinstance(val, dict):
            return val.get('name')
        return None

    df[dict_json_col] = df[dict_json_col].apply(safe_parse).apply(extract_collection_name)

    return df


def convert_column_types(df):
    """
    Converts column datatypes:
    - 'budget', 'id', 'popularity' → numeric (invalid values become NaN)
    - 'release_date' → datetime (invalid values become NaT)
    """
    # Convert to numeric
    cols_to_numeric = ['budget', 'id', 'popularity']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert to datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    return df


def clean_unrealistic_values(df):
    """
    Cleans unrealistic values in the movie dataset.
    """
    import numpy as np

    # Replace 0 with NaN
    for col in ['budget', 'revenue', 'runtime']:
        df[col] = df[col].replace(0, np.nan)

    # Convert to millions
    for col in ['budget', 'revenue']:
        df[col] = df[col] / 1_000_000

    # vote_count = 0 → vote_average = NaN
    df.loc[df['vote_count'] == 0, 'vote_average'] = np.nan

    # Replace placeholders in text fields
    placeholders = ['No Data', 'no data', '', 'nan', 'N/A', 'None']
    for col in ['overview', 'tagline']:
        df[col] = df[col].replace(placeholders, np.nan)

    return df


def clean_unrealistic_values(df):
    """
    Cleans unrealistic values in the movie dataset.
    """
    import numpy as np

    # Replace 0 with NaN
    for col in ['budget', 'revenue', 'runtime']:
        df[col] = df[col].replace(0, np.nan)

    # Convert to millions
    for col in ['budget', 'revenue']:
        df[col] = df[col] / 1_000_000

    # vote_count = 0 → vote_average = NaN
    df.loc[df['vote_count'] == 0, 'vote_average'] = np.nan

    # Replace placeholders in text fields
    placeholders = ['No Data', 'no data', '', 'nan', 'N/A', 'None']
    for col in ['overview', 'tagline']:
        df[col] = df[col].replace(placeholders, np.nan)

    return df

def remove_duplicates_and_missing_ids(df):
    """
    Removes duplicate movies based on 'id' and 'title',
    and drops rows with missing 'id' or 'title'.
    """
    # Drop duplicates
    df = df.drop_duplicates(subset=['id', 'title'])

    # Drop rows with missing ID or title
    df = df.dropna(subset=['id', 'title'])

    return df

def filter_valid_and_released_movies(df, min_non_null_cols=10):
    """
    Keeps rows with at least `min_non_null_cols` non-NaN values,
    filters to only 'Released' movies, and drops the 'status' column.
    """
    # Keep rows with enough non-NaN values
    df = df[df.notna().sum(axis=1) >= min_non_null_cols]

    # Filter only released movies
    df = df[df['status'] == 'Released']

    # Drop the 'status' column
    df = df.drop(columns=['status'])

    return df

def rename_cleaned_columns(df):
    """
    Renames cleaned/intermediate column names to match final project format.
    """
    df = df.rename(columns={
        'collection_name': 'belongs_to_collection',
        'genres_clean': 'genres',
        'spoken_languages_clean': 'spoken_languages',
        'production_countries_clean': 'production_countries',
        'production_companies_clean': 'production_companies',
        'budget': 'budget_musd',
        'revenue': 'revenue_musd'
    })
    return df


import pandas as pd
import ast

def add_cast_crew_columns(df):
    """
    Processes the 'credits' column:
    - Parses JSON-like strings to dicts
    - Extracts:
        * 'cast' (top 5 names, separated by '|')
        * 'cast_size' (total number of cast members)
        * 'director' (first director found)
        * 'crew_size' (total number of crew members)
    - Drops the original 'credits' column
    Returns updated DataFrame
    """

    # Safely parse the 'credits' column if still a string
    df['credits'] = df['credits'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Function to extract fields from one row
    def extract_cast_crew_info(credits_dict):
        if not isinstance(credits_dict, dict):
            return pd.Series([None, None, None, None], index=['cast', 'cast_size', 'director', 'crew_size'])

        cast_list = credits_dict.get('cast', [])
        crew_list = credits_dict.get('crew', [])

        top_cast = [member['name'] for member in cast_list[:5] if 'name' in member]
        directors = [member['name'] for member in crew_list if member.get('job') == 'Director']

        return pd.Series([
            '|'.join(top_cast),
            len(cast_list),
            directors[0] if directors else None,
            len(crew_list)
        ], index=['cast', 'cast_size', 'director', 'crew_size'])

    # Apply to DataFrame and merge results
    cast_crew_df = df['credits'].apply(extract_cast_crew_info)
    df = pd.concat([df, cast_crew_df], axis=1)

    # Drop original 'credits' column
    df.drop(columns=['credits'], inplace=True)

    return df


def reorder_columns(df):
    """
    Reorders the DataFrame columns to match the final project structure.
    Drops any missing columns from the order if they don't exist in the DataFrame.
    """
    column_order = [
        'id', 'title', 'tagline', 'release_date', 'genres', 'belongs_to_collection',
        'original_language', 'budget_musd', 'revenue_musd', 'production_companies',
        'production_countries', 'vote_count', 'vote_average', 'popularity', 'runtime',
        'overview', 'spoken_languages', 'poster_path', 'cast', 'cast_size', 'director', 'crew_size'
    ]

    # Only keep columns that actually exist in df
    available_columns = [col for col in column_order if col in df.columns]
    
    return df[available_columns]


def reset_df_index(df):
    """
    Resets the index of the DataFrame and drops the old index.
    """
    return df.reset_index(drop=True)






