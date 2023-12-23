import pandas as pd
from sklearn.model_selection import train_test_split
import utils.preprocess
from tqdm import tqdm

tqdm.pandas()


'''
Input dataset is expected to have the following data fields after downloading from Twitter API:
- tweet_id, country_code, city, tweet_text, tweet_lang, user_name, user_description, user_location
'''

df_data = pd.read_csv('data/DOWNLOADED_DATA.csv')


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


'''
Employ stratified sampling to deal with imbalanced sample.
'''

df_train, df_val = train_test_split(df_data, test_size=0.2, random_state=42, stratify=df_data.city)
df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=42, stratify=df_val.city)


'''
Partition the sample into three subsets: training set (80%), validation set (10%), and test set (10%).
Save the cleaned data to parquet and json files.
'''

df_train.reset_index(drop=True).to_parquet('data/prep_3_train_split_801010.parquet')
df_val.reset_index(drop=True).to_parquet('data/prep_3_val_split_801010.parquet')
df_test.reset_index(drop=True).to_parquet('data/prep_3_test_split_801010.parquet')

df_train.reset_index(drop=True).to_json('data/prep_3_train_split_801010.json', orient='records', lines=True)
df_val.reset_index(drop=True).to_json('data/prep_3_val_split_801010.json', orient='records', lines=True)
df_test.reset_index(drop=True).to_json('data/prep_3_test_split_801010.json', orient='records', lines=True)