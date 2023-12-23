import pandas as pd
import utils.preprocess
from tqdm import tqdm

tqdm.pandas()


'''
Input dataset is expected to have the following data fields after downloading from Twitter API:
- user_id, tweet_id, tweet_text, tweet_lang, user_name, user_description, user_location
'''

df_data = pd.read_csv('data/DOWNLOADED_DATA.csv')

# Store the tweet count for each user, will be used in b3_finalise_output.py later
df_count = df_data.groupby('user_id').agg({'tweet_id': 'count'})
df_count = df_count.rename(columns={'tweet_id': 'tweet_count'})
df_count.to_csv('prediction_dataset_tweet_count.csv', index=True, sep=',')


'''
Clean the text data by:
 (i) removing links, user names, punctuation, extra spaces,
 (ii) separating emoticons (using Python emoji package 2.2.0 on https://pypi.org/project/emoji/), and
 (iii) making all text lower case.

Refer to utils\preprocess.py for the clean_text_series() function.
'''

df_data['cleaned_text'] = utils.preprocess.clean_text_series(df_data['tweet_text'], mode='text')
df_data['cleaned_user_name'] = utils.preprocess.clean_text_series(df_data['user_name'], mode='text')
df_data['cleaned_user_description'] = utils.preprocess.clean_text_series(df_data['user_description'], mode='text')
df_data['cleaned_user_location'] = utils.preprocess.clean_text_series(df_data['user_location'], mode='loc')


# Export the cleaned prediction dataset
cat2str = {'tweet_lang': 'str', 'geo_country_code': 'str'}
df_data.astype(cat2str).replace({None: ''}).fillna('').to_json(f'data/authlist_to_infer_cleaned.json', orient='records', lines=True)