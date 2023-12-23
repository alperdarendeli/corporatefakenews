import pandas as pd
import re, emoji


def clean_text(text: str, mode: str = None) -> str:
    '''
    Performs text cleaning to separate and standardize words
    '''
    if len(text) == 0: return ''
    if mode in ['text', 'loc']:
        text = re.sub(r'http[s]*://[^\s]+', '', text)                                       # remove links
        text = re.sub(r'@[\w]+', '', text)                                                  # remove usernames
        text = re.sub(r'[,.±©®-×÷]', ' ', text)                                             # remove punctuation 1
        text = re.sub(r'[!"#$%&\'()*/\\+:;<=>?@\[\]^_`{|}~—“”]', ' ', text)                 # remove punctuation 2
        text = emoji.emojize(re.sub(u'(:[\w]+:)', r' \1 ', emoji.demojize(text)))           # separate emoji
        text = re.sub(r'[\s]+', ' ', text)                                                  # extra spaces
    if mode == 'loc':
        text = re.sub(r'[0-9]+', '', text)                                                  # numbers
    return text.strip().lower()                                                             # lowercase, extra spaces


def clean_text_series(text: pd.Series, mode: str = None) -> pd.Series:
    '''
    Performs text cleaning to separate and standardize words, optimised for pd.Series
    '''
    text = text.fillna('')
    if mode in ['text', 'loc']:
        text = text.str.replace(r'http[s]*://[^\s]+', '', regex=True)                       # remove links
        text = text.str.replace(r'@[\w]+', '', regex=True)                                  # remove usernames
        text = text.str.replace(r'[,.±©®-×÷]', ' ', regex=True)                             # remove punctuation 1
        text = text.str.replace(r'[!"#$%&\'()*/\\+:;<=>?@\[\]^_`{|}~—“”]', ' ', regex=True) # remove punctuation 2
        text = text.progress_apply(emoji.demojize)                                          # separate emoji
        text = text.str.replace(u'(:[\w]+:)', r' \1 ', regex=True)
        text = text.progress_apply(emoji.emojize)
        text = text.str.replace(r'[\s]+', ' ', regex=True)                                  # extra spaces
    if mode == 'loc':
        text = text.str.replace(r'[0-9]+', '', regex=True)                                  # numbers
    return text.str.strip().str.lower()                                                     # lowercase, extra spaces


if __name__ == '__main__':

    sample_text = 'testing @maurieast te quieroooo البيت https://t.co/AP (@ 東京駅 in 千代田 🤱🏽❣️❣️'
    print('Before ::', sample_text)
    print('After  ::', clean_text(sample_text, mode='text'))