import datasets
import transformers
import pandas as pd
from tqdm import tqdm
import argparse, os

from utils.dataset import TCD_Selector
tqdm.pandas()

'''
This scripts builts the tokenizer based on the combined feature and saves it in the tokenizer/features_{feat}/ folder.
'''


def main(args):

    print(f'>> Selected features: {",".join(args.feat)}')
    print('>> Loading training dataset ...')
    df_train = pd.read_parquet('data/prep_3_train_split_801010.parquet')
    os.makedirs('tokenizer/combined_cache', exist_ok=True)

    for feat in args.feat:


        '''
        Concatenate the features as follows:
          [TEXT] <cleaned_text> [LANG] <tweet_lang> [LOC] <cleaned_user_declared_ location> 
          [DESC] <cleaned_user_description> [NAME] <cleaned_user_name>

        Refer to utils\dataset.py for the TCD_Selector class which handles the concatenation of features
        based on the feature combination selected with the _concat_features(data) method.
        '''

        print(f'>> [feat={feat}] Generating combined text ...')
        combined_cache = f'./tokenizer/combined_cache/prep_3_train_split_801010_features_{feat}.txt'

        # If the combined text file already exists, load it from the cache
        # Otherwise, generate a txt file of the combined features and save it as cache
        if not os.path.exists(combined_cache):
            combined_text = df_train.progress_apply(TCD_Selector[feat]._concat_features, axis=1)
            with open(combined_cache, 'w') as f:
                for line in combined_text:
                    f.write(line)
                    f.write('\n')

        # Load the text file using huggingface datasets library
        print(f'>> [feat={feat}] Building tokenizer ...')
        dataset = datasets.load_dataset('text', data_files=combined_cache)
        training_corpus = (dataset['train'][i:i+1000]['text'] for i in range(0, len(dataset['train']), 1000))


        '''
        Employ BertTokenizerFast (https://huggingface.co/bert-base-multilingual-uncased) to tokenize Twitter features.
        '''

        # Build the tokenizer with a vocabulary size of 100000
        tokenizer_old = transformers.AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', use_fast=True)
        tokenizer_new = tokenizer_old.train_new_from_iterator(
            text_iterator=training_corpus,
            vocab_size=100000,
            length=len(dataset['train']),
            new_special_tokens=['[TEXT]', '[LANG]', '[LOC]', '[DESC]', '[NAME]', '[BLANK]']
        )

        # Save the tokenizer
        tokenizer_new.save_pretrained(f'./tokenizer/features_{feat}/')
        print(f'>> [feat={feat}] Tokenizer built successfully')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-feat', type=str, required=True, choices=TCD_Selector.keys(), nargs='+', help='tweet features')
    args = parser.parse_args()

    main(args)
