# %%
import json
import pandas as pd
import numpy as np
import re
import psycopg2
import time
from config import db_password
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# %%
file = 'C:/Users/rahul/Desktop/Class Folder/Movies-ETL'


# %%
def wiki_movies(file):
    wiki_movies_raw = json.load(open(f'{file}/wikipedia.movies.json', mode='r'))
    wiki_movies = [movie for movie in wiki_movies_raw
                   if ('Director' in movie or 'Directed by' in movie)
                       and 'imdb_link' in movie
                       and 'No. of episodes' not in movie]

    # After looping through every key, add the alternative titles dict to the movie object.
    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune–Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles
        return movie
        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')

        return movie

    # Run list comprehension to clean wiki_movies and create wiki_movies_df
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    wiki_movies_df = pd.DataFrame(clean_movies)
    
    try:
        # Drop duplicates of IMDb ID's by using drop_duplicates() method.
        wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
        wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
    
    except Continue as c:
        print(c)

    # List of columns that have less than 90% null values and use them to trim down the dataset.
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    #  Select rows on box office data that have date by dropping the ones with missing values.
    box_office = wiki_movies_df['Box office'].dropna()

    box_office[box_office.map(lambda x: type(x) != str)]

    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


    # Some values have spaces in between the dollar sign and the number.
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

    #  Some values are given as a range.
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    #  extract only the parts of the strings that match.
    box_office.str.extract(f'({form_one}|{form_two})')

    #  function to turn the extracted values into a numeric value.
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " million"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a million
            value = float(s) * 10**6

            # return value
            return value

        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " billion"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a billion
            value = float(s) * 10**9

            # return value
            return value

        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

            # remove dollar sign and commas
            s = re.sub('\$|,','', s)

            # convert to float
            value = float(s)

            # return value
            return value

        # otherwise, return NaN
        else:
            return np.nan

    #  extract the values from box_office using str.extract
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    # Drop Box Office column.
    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    #  Create a budget variable.
    budget = wiki_movies_df['Budget'].dropna()

    # Convert any lists to strings
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

    #  remove any values between a dollar sign and a hyphen
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)

    #  Remove the citation references
    budget = budget.str.replace(r'\[\d+\]\s*', '')

    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    #  drop the original Budget column.
    wiki_movies_df.drop('Budget', axis=1, inplace=True)

    #  Parsing the release date
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    #  Parsing using regular expressions.
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'

    #  extract the dates
    release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)

    #  use the built-in to_datetime() method in Pandas to parse dates.
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    #  make a variable that holds the non-null values of Release date in the DataFrame, converting lists to strings
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]

    #  make this more general by only marking the beginning of the string, and accepting other abbreviations of “minutes” by only searching up to the letter “m.”
    running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()

    #  The remaining 17 follow.
    running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]

    #  only want to extract digits, and we want to allow for both possible patterns
    # capture groups around the \d instances as well as add an alternating character
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

    #  DataFrame is all strings, hence need to convert them to numeric values.
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

    #  apply a function that will convert the hour capture groups and minute capture groups to minutes
    #  if the pure minutes capture group is zero, and save the output to wiki_movies_df:
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

    #  drop Running time from the dataset
    wiki_movies_df.drop('Running time', axis=1, inplace=True)
    
    return wiki_movies_df


# %%
def kaggle(file):
    kaggle_metadata = pd.read_csv(f'{file}/movies_metadata.csv')
    #  remove the bad data
    kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]

    #  keep rows where the adult column is False, and then drop the adult column
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

    #  Convert data types
    kaggle_metadata['video'] == 'True'

    #  assign the boolean column above back to video.
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')

    #  convert release_date to datetime
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])
    
    return kaggle_metadata


# %%
def merge(wiki_movies_df, kaggle_metadata):
    
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

    #  drop the row containing The Holiday in the Wikipedia data as it got merged with From Here to Eternity
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

    # convert the lists in Language to tuples so that the value_counts() method will work
    movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)

    # For the Kaggle data, there are no lists, so we can just run value_counts().
    movies_df['original_language'].value_counts(dropna=False)

    # drop the title_wiki, release_date_wiki, Language columns
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language'],  inplace = True)

    #  make a function that fills in missing data for a column pair and then drops the redundant column
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column], axis=1)
        df.drop(columns=wiki_column, inplace=True)

    # run the function for the three column pairs that we decided to fill in zeros
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    # convert lists to tuples for value_counts() to work
    for col in movies_df.columns:
        lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
        value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
        num_values = len(value_counts)
        if num_values == 1:
            print(col)

    movies_df['video'].value_counts(dropna=False)

    # Rearrange columns to make sense for any reader
    column_titles = ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link','runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count', 'genres','original_language','overview','spoken_languages','Country', 'production_companies','production_countries','Distributor','Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on']
    movies_df = movies_df.reindex(columns=column_titles)

    #  rename the columns to be consistent
    movies_df.rename({'id':'kaggle_id',
                      'title_kaggle':'title',
                      'url':'wikipedia_url',
                      'budget_kaggle':'budget',
                      'release_date_kaggle':'release_date',
                      'Country':'country',
                      'Distributor':'distributor',
                      'Producer(s)':'producers',
                      'Director':'director',
                      'Starring':'starring',
                      'Cinematography':'cinematography',
                      'Editor(s)':'editors',
                      'Writer(s)':'writers',
                      'Composer(s)':'composers',
                      'Based on':'based_on'
                     }, axis='columns', inplace=True)
       
    return movies_df


# %%
def ratings(file, movies_df):
    
    ratings = pd.read_csv(f'{file}/ratings.csv')
    #  look at the ratings data by setting the null counts to true.
    ratings.info(null_counts=True)

    pd.to_datetime(ratings['timestamp'], unit='s')

    #  Assign the timestamp to columns
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    #  count movieID and ratings columns
    #  rename the “userId” column to “count.”
    #  pivot this data so that movieId is the index, the columns will be all the rating values,
    #  and the rows will be the counts for each rating value.
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                     .rename({'userId':'count'}, axis=1)                     .pivot(index='movieId',columns='rating', values='count')

    #  rename the columns so they’re easier to understand
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    #  Fill missing values on ratings.
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)
    
    return


# %%
def export(file, movies_df):
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
    engine = create_engine(db_string)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    session.execute('''TRUNCATE TABLE movies''')
    session.commit()
    session.close()
    
    movies_df.to_sql(name='movies', con=engine, if_exists='append')

    
    rows_imported = 0

    # get the start_time from time.time()
    start_time = time.time()

    for data in pd.read_csv(f'{file}/ratings.csv', chunksize=1000000):
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)

        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')
        
        return


# %%
def Challenge(file):

    wiki_movies(file)
    wiki_data = wiki_movies(file)
    
    kaggle(file)
    kaggle_data = kaggle(file)   
    
    merge(wiki_data, kaggle_data)
    merge_data = merge(wiki_data, kaggle_data)
    
    ratings(file, merge_data)
    
    export(file, merge_data)
    
    return


# %%
Challenge(file)


