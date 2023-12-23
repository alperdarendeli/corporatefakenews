import tensorflow as tf
import transformers
import json
from tqdm import tqdm

# Add root to sys path
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import utils.preprocess
from typing import Dict


def preprocess_from_dict(data: Dict) -> str:
    '''
    Pre-process the text by cleaning and concatenating them
    '''
    c_text = utils.preprocess.clean_text(data['tweet_text'], mode='text')
    c_user_name = utils.preprocess.clean_text(data['user_name'], mode='text')
    c_user_description = utils.preprocess.clean_text(data['user_description'], mode='text')
    c_user_location = utils.preprocess.clean_text(data['user_location'], mode='loc')
    return ' '.join([
        '[TEXT] ', c_text if len(c_text) else '[BLANK]',
        '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]',
        '[LOC]', c_user_location if len(c_user_location) else '[BLANK]',
        '[DESC]', c_user_description if len(c_user_description) else '[BLANK]',
        '[NAME]', c_user_name if len(c_user_name) else '[BLANK]'
    ])


class TweetCleanDataset:

    '''
    Tweet dataset loaded from a json file where each line is an entry, loops to the start when
    the end of file is reached. Text is concatenated and tokenized, functioning as a generator.
    :: Already bulk cleaned in another script i.e. has fields with cleaned_ prefix.
    :: Set eval=True for evaluation mode without y labels.
    :: Specify start_index to start from that entry with that index.
    :: Factory method pattern is used here, call the corresponding class based on the features required.
    '''

    def __init__(self, json_filepath: str, batch_size: int, features: str, eval: bool = False, start_index: int = 0):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'tokenizer/features_{features}/', use_fast=True)
        self.eval, self.batch_size = eval, batch_size
        self.pretokenized_texts, self.ids = [None] * batch_size, [None] * batch_size
        self.total_lines, self.current_line = 0, start_index
        self.num_batches  = int(self.total_lines / self.batch_size + 0.5)
        self.file = open(json_filepath, 'r')
        for _ in tqdm(self.file, desc=f'Loading entries from {json_filepath}'):
            self.total_lines += 1
        self.file.seek(0)
        for _ in range(self.current_line):
            self.file.readline()

    def __len__(self):
        return int(self.total_lines / self.batch_size + 0.5)

    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]',
            '[LOC]', data['cleaned_user_location'] if len(data['cleaned_user_location']) else '[BLANK]',
            '[DESC]', data['cleaned_user_description'] if len(data['cleaned_user_description']) else '[BLANK]',
            '[NAME]', data['cleaned_user_name'] if len(data['cleaned_user_name']) else '[BLANK]'
        ])

    def _get_one_x_y(self):
        if self.current_line >= self.total_lines:
            self.file.seek(0)
            self.current_line = 0
        data = json.loads(self.file.readline())
        self.current_line += 1
        return self._concat_features(data), self.current_line-1 if self.eval else data['city_id']

    def get_batch(self):
        for i in range(self.batch_size):
            self.pretokenized_texts[i], self.ids[i] = self._get_one_x_y()
        x_batch = self.tokenizer(self.pretokenized_texts, max_length=256, padding='max_length', truncation=True)['input_ids']
        return tf.constant(x_batch), tf.constant(self.ids)

    def __del__(self):
        self.file.close()


class TweetRawDataset:

    '''
    Tweet dataset loaded from a json file where each line is an entry, loops to the start when
    the end of file is reached. Text is concatenated and tokenized, functioning as a generator.
    :: Text is cleaned on the go with the preprocess_from_dict function.
    :: Set eval=True for evaluation mode without y labels.
    :: Specify start_index to start from that entry with that index.
    '''

    def __init__(self, json_filepath: str, batch_size: int, features: str, eval: bool = False, start_index: int = 0):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'tokenizer/features_{features}/', use_fast=True)
        self.eval, self.batch_size = eval, batch_size
        self.pretokenized_texts, self.ids = [None] * batch_size, [None] * batch_size
        self.total_lines, self.current_line = 0, start_index
        self.num_batches  = int(self.total_lines / self.batch_size + 0.5)
        self.file = open(json_filepath, 'r')
        for _ in tqdm(self.file, desc=f'Loading entries from {json_filepath}'):
            self.total_lines += 1
        self.file.seek(0)
        for _ in range(self.current_line):
            self.file.readline()

    def __len__(self):
        return int(self.total_lines / self.batch_size + 0.5)

    def _get_one_x_y(self):
        if self.current_line >= self.total_lines:
            self.file.seek(0)
            self.current_line = 0
        data = json.loads(self.file.readline())
        self.current_line += 1
        return preprocess_from_dict(data), self.current_line-1 if self.eval else data['city_id']

    def get_batch(self):
        for i in range(self.batch_size):
            self.pretokenized_texts[i], self.ids[i] = self._get_one_x_y()
        x_batch = self.tokenizer(self.pretokenized_texts, max_length=256, padding='max_length', truncation=True)['input_ids']
        return tf.constant(x_batch), tf.constant(self.ids)

    def __del__(self):
        self.file.close()


# ----- FEATURE EXPERIMENTS -----

'''Use TCD_Selector to select the desired preprocessed method for the features'''

class TCD_Text(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]'
        ])

class TCD_Lang(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]'
        ])

class TCD_Loc(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[LOC]', data['cleaned_user_location'] if len(data['cleaned_user_location']) else '[BLANK]'
        ])

class TCD_Desc(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[DESC]', data['cleaned_user_description'] if len(data['cleaned_user_description']) else '[BLANK]'
        ])

class TCD_Name(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[NAME]', data['cleaned_user_name'] if len(data['cleaned_user_name']) else '[BLANK]'
        ])

class TCD_TextLang(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]'
        ])

class TCD_TextLoc(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LOC]', data['cleaned_user_location'] if len(data['cleaned_user_location']) else '[BLANK]'
        ])

class TCD_TextDesc(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[DESC]', data['cleaned_user_description'] if len(data['cleaned_user_description']) else '[BLANK]'
        ])

class TCD_TextName(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[NAME]', data['cleaned_user_name'] if len(data['cleaned_user_name']) else '[BLANK]'
        ])

class TCD_TextLangLoc(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]',
            '[LOC]', data['cleaned_user_location'] if len(data['cleaned_user_location']) else '[BLANK]'
        ])

class TCD_TextLangDesc(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]',
            '[DESC]', data['cleaned_user_description'] if len(data['cleaned_user_description']) else '[BLANK]'
        ])

class TCD_TextLangName(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]',
            '[NAME]', data['cleaned_user_name'] if len(data['cleaned_user_name']) else '[BLANK]'
        ])

class TCD_TextLocDesc(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LOC]', data['cleaned_user_location'] if len(data['cleaned_user_location']) else '[BLANK]',
            '[DESC]', data['cleaned_user_description'] if len(data['cleaned_user_description']) else '[BLANK]'
        ])

class TCD_TextLocName(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LOC]', data['cleaned_user_location'] if len(data['cleaned_user_location']) else '[BLANK]',
            '[NAME]', data['cleaned_user_name'] if len(data['cleaned_user_name']) else '[BLANK]'
        ])

class TCD_TextDescName(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[DESC]', data['cleaned_user_description'] if len(data['cleaned_user_description']) else '[BLANK]',
            '[NAME]', data['cleaned_user_name'] if len(data['cleaned_user_name']) else '[BLANK]'
        ])

class TCD_TextLangLocDesc(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]',
            '[LOC]', data['cleaned_user_location'] if len(data['cleaned_user_location']) else '[BLANK]',
            '[DESC]', data['cleaned_user_description'] if len(data['cleaned_user_description']) else '[BLANK]'
        ])

class TCD_TextLangLocName(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]',
            '[LOC]', data['cleaned_user_location'] if len(data['cleaned_user_location']) else '[BLANK]',
            '[NAME]', data['cleaned_user_name'] if len(data['cleaned_user_name']) else '[BLANK]'
        ])

class TCD_TextLangDescName(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LANG]', data['tweet_lang'] if len(data['tweet_lang']) else '[BLANK]',
            '[DESC]', data['cleaned_user_description'] if len(data['cleaned_user_description']) else '[BLANK]',
            '[NAME]', data['cleaned_user_name'] if len(data['cleaned_user_name']) else '[BLANK]'
        ])

class TCD_TextLocDescName(TweetCleanDataset):
    @staticmethod
    def _concat_features(data):
        return ' '.join([
            '[TEXT]', data['cleaned_text'] if len(data['cleaned_text']) else '[BLANK]',
            '[LOC]', data['cleaned_user_location'] if len(data['cleaned_user_location']) else '[BLANK]',
            '[DESC]', data['cleaned_user_description'] if len(data['cleaned_user_description']) else '[BLANK]',
            '[NAME]', data['cleaned_user_name'] if len(data['cleaned_user_name']) else '[BLANK]'
        ])

TCD_Selector = {
    'all': TweetCleanDataset,
    'comp1': TCD_Text,
    'comp2': TCD_Lang,
    'comp3': TCD_Loc,
    'comp4': TCD_Desc,
    'comp5': TCD_Name,
    'comp12': TCD_TextLang,
    'comp13': TCD_TextLoc,
    'comp14': TCD_TextDesc,
    'comp15': TCD_TextName,
    'comp123': TCD_TextLangLoc,
    'comp124': TCD_TextLangDesc,
    'comp125': TCD_TextLangName,
    'comp134': TCD_TextLocDesc,
    'comp135': TCD_TextLocName,
    'comp145': TCD_TextDescName,
    'comp1234': TCD_TextLangLocDesc,
    'comp1235': TCD_TextLangLocName,
    'comp1245': TCD_TextLangDescName,
    'comp1345': TCD_TextLocDescName,
}


# ----- UNIT TESTS -----

if __name__ == '__main__':

    datagen1 = TCD_Selector['all'](json_filepath='data/prep_3_test.json', batch_size=64)
    datagen2 = TCD_Selector['comp12'](json_filepath='data/prep_3_test.json', batch_size=64)
    
    print('\n[UNIT TEST #1]  tweet clean dataset selector')
    x_batch, y_batch = datagen1.get_batch()
    print('all-index1 ::', datagen1.tokenizer.batch_decode(x_batch)[0][:200])
    print('all-index2 ::', datagen1.tokenizer.batch_decode(x_batch)[1][:200])
    x_batch, y_batch = datagen2.get_batch()
    print('comp12-index1 ::', datagen2.tokenizer.batch_decode(x_batch)[0][:200])
    print('comp12-index2 ::', datagen2.tokenizer.batch_decode(x_batch)[1][:200])
    print('x,y information ::', x_batch.shape, x_batch.dtype, y_batch.shape, y_batch.dtype)

    print('\n[UNIT TEST #2]  cycle back to start')
    for _ in tqdm(range(len(datagen1)-1)): x_batch, y_batch = datagen1.get_batch()
    print('all-index1 ::', datagen1.tokenizer.batch_decode(x_batch)[32][:200])
    print('all-index2 ::', datagen1.tokenizer.batch_decode(x_batch)[33][:200])


    # self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', use_fast=True)
    # self.tokenizer.add_tokens(['[TEXT]', '[LANG]', '[LOC]', '[DESC]', '[NAME]', '[BLANK]'])
    # self.vocab_size = len(self.tokenizer.vocab.keys())