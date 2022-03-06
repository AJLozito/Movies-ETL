#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To start, we need to import three dependencies:

# JSON library to extract the Wikipedia data
# Pandas library to create DataFrames
# NumPy library for converting data types


import json
import pandas as pd
import numpy as np


# In[2]:


file_dir = r"C:\Users\ajloz\OneDrive\Desktop\UCF Bootcamp\Module 8\Data\wikipedia-movies.json"


# In[3]:


with open(file_dir, mode='r') as file:
    wiki_movies_raw = json.load(file)


# In[4]:


len(wiki_movies_raw)


# In[5]:


# First 5 records
wiki_movies_raw[:5]


# In[6]:


# Last 5 records
wiki_movies_raw[-5:]


# In[7]:


# Some records in the middle
wiki_movies_raw[3600:3605]


# In[8]:


kaggle_metadata = r"C:\Users\ajloz\OneDrive\Desktop\UCF Bootcamp\Module 8\Data\movies_metadata.csv"


# In[9]:


kaggle_metadata_df = pd.read_csv(kaggle_metadata, low_memory=False)


# In[10]:


ratings = r"C:\Users\ajloz\OneDrive\Desktop\UCF Bootcamp\Module 8\Data\ratings.csv"


# In[11]:


ratings_df = pd.read_csv(ratings, low_memory=False)


# In[12]:


wiki_movies_df = pd.DataFrame(wiki_movies_raw)


# In[13]:


wiki_movies_df.head()


# In[14]:


wiki_movies_df.columns.tolist()


# In[15]:


wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie]
len(wiki_movies)


# In[16]:


wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]


# In[17]:


x = 'global value'

def foo():
    x = 'local value'
    print(x)

foo()
print(x)


# In[18]:


my_list = [1,2,3]
def append_four(x):
    x.append(4)
append_four(my_list)
print(my_list)


# In[19]:


square = lambda x: x * x
square(5)


# In[20]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    return movie


# In[21]:


wiki_movies_df[wiki_movies_df['Arabic'].notnull()]


# In[22]:


wiki_movies_df[wiki_movies_df['Arabic'].notnull()]['url']


# In[23]:


sorted(wiki_movies_df.columns.tolist())


# In[24]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    return movie


# In[25]:


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


# In[26]:


clean_movies = [clean_movie(movie) for movie in wiki_movies]


# In[27]:


wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())


# In[28]:


def change_column_name(old_name, new_name):
    if old_name in movie:
        movie[new_name] = movie.pop(old_name)


# In[29]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

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


# In[30]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

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


# In[31]:


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')


# In[32]:


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()


# In[33]:


[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]


# In[34]:


[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]


# In[35]:


wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]


# In[36]:


box_office = wiki_movies_df['Box office'].dropna()


# In[37]:


def is_not_a_string(x):
    return type(x) != str


# In[38]:


box_office[box_office.map(is_not_a_string)]


# In[39]:


lambda arguments: expression


# In[40]:


lambda x: type(x) != str


# In[41]:


box_office[box_office.map(lambda x: type(x) != str)]


# In[42]:


some_list = ['One','Two','Three']
'Mississippi'.join(some_list)


# In[43]:


box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[44]:


import re


# In[45]:


form_one = r'\$\d+\.?\d*\s*[mb]illion'


# In[46]:


box_office.str.contains(form_one, flags=re.IGNORECASE, na=False).sum()


# In[47]:


form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE, na=False).sum()


# In[48]:


matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE, na=False)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE, na=False)


# In[49]:


box_office[~matches_form_one & ~matches_form_two]


# In[50]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illion'
form_two = r'\$\s*\d{1,3}(?:,\d{3})+'


# In[51]:


form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+'


# In[52]:


form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'


# In[53]:


box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[54]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'


# In[55]:


box_office.str.extract(f'({form_one}|{form_two})')


# In[56]:


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


# In[57]:


wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[58]:


wiki_movies_df.drop('Box office', axis=1, inplace=True)


# In[59]:


budget = wiki_movies_df['Budget'].dropna()


# In[60]:


budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)


# In[61]:


budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[62]:


matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE, na=False)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE, na=False)
budget[~matches_form_one & ~matches_form_two]


# In[63]:


budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]


# In[64]:


wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[65]:


wiki_movies_df.drop('Budget', axis=1, inplace=True)


# In[66]:


release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[67]:


date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]?\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[0123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


# In[68]:


release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)


# In[69]:


wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


# In[70]:


running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[71]:


running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE, na=False).sum()


# In[72]:


running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE, na=False) != True]


# In[73]:


running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE, na=False).sum()


# In[74]:


running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE, na=False) != True]


# In[75]:


running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[76]:


running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[77]:


wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# In[78]:


wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[79]:


kaggle_metadata_df.dtypes


# In[80]:


kaggle_metadata_df['adult'].value_counts()


# In[81]:


kaggle_metadata_df[~kaggle_metadata_df['adult'].isin(['True','False'])]


# In[82]:


kaggle_metadata_df = kaggle_metadata_df[kaggle_metadata_df['adult'] == 'False'].drop('adult',axis='columns')


# In[83]:


kaggle_metadata_df['video'].value_counts()


# In[84]:


kaggle_metadata_df['video'] == 'True'


# In[85]:


kaggle_metadata_df['video'] = kaggle_metadata_df['video'] == 'True'


# In[86]:


kaggle_metadata_df['budget'] = kaggle_metadata_df['budget'].astype(int)
kaggle_metadata_df['id'] = pd.to_numeric(kaggle_metadata_df['id'], errors='raise')
kaggle_metadata_df['popularity'] = pd.to_numeric(kaggle_metadata_df['popularity'], errors='raise')


# In[87]:


kaggle_metadata_df['release_date'] = pd.to_datetime(kaggle_metadata_df['release_date'])


# In[88]:


ratings_df.info(null_counts=True)


# In[89]:


pd.to_datetime(ratings_df['timestamp'], unit='s')


# In[90]:


ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')


# In[91]:


pd.options.display.float_format = '{:20,.2f}'.format
ratings_df['rating'].plot(kind='hist')
ratings_df['rating'].describe()


# In[92]:


movies_df = pd.merge(wiki_movies_df, kaggle_metadata_df, on='imdb_id', suffixes=['_wiki','_kaggle'])


# In[93]:


movies_df[['title_wiki','title_kaggle']]


# In[94]:


movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]


# In[95]:


# Show any rows where title_kaggle is empty
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


# In[96]:


movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# In[97]:


movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# In[98]:


movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')


# In[99]:


movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')


# In[100]:


movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[101]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


# In[102]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index


# In[103]:


movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[104]:


movies_df[movies_df['release_date_wiki'].isnull()]


# In[105]:


movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[106]:


movies_df['original_language'].value_counts(dropna=False)


# In[107]:


def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


# In[108]:


fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df


# In[109]:


for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)


# In[110]:


movies_df['video'].value_counts(dropna=False)


# In[111]:


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


# In[112]:


rating_counts = ratings_df.groupby(['movieId','rating'], as_index=False).count()


# In[113]:


rating_counts = ratings_df.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)


# In[114]:


rating_counts = ratings_df.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')


# In[115]:


rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


# In[116]:


movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')


# In[117]:


movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


# In[118]:


from sqlalchemy import create_engine


# In[119]:


"postgresql://[postgres]:[Knightro101]@[PostgreSQL]:[Databases]/[movie_data]"


# In[120]:


db_password = 'Knightro101!'


# In[121]:


from config import db_password


# In[ ]:


db_string = f"postgresql://postgres:{db_password}@127.0.0.1:5432/movie_data"


# In[ ]:


engine = create_engine(db_string)


# In[ ]:


movies_df.to_sql(name='movies', con=engine)


# In[ ]:


rows_imported = 0
for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):

    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    print(f'Done.')


# In[ ]:


import time


# In[ ]:


rows_imported = 0
# get the start_time from time.time()
start_time = time.time()
for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done. {time.time() - start_time} total seconds elapsed')

