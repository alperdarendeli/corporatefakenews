import pandas as pd
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm
import glob, re

tqdm.pandas()


class_mapper = pd.read_csv('utils/cities_mapper_new.csv').iloc[:, 2:]

def main(SEED):

    '''
    The predicted tweet probabilities are aggregated by user and the argmax class is taken as the final prediction.
    '''
    
    for logit_file in sorted(glob.glob(f'seeded-{SEED}/results/authlist/logits_authlist_*')):
        
        model_name = re.findall('(ndim_[\d]+_nlstm_[\d]+)', logit_file)[0]
        logits = sps.load_npz(logit_file).astype(np.uint16)

        df_pred_user = pd.read_csv('prediction_dataset_tweet_count.csv')
        cum_sum = [0] + df_pred_user['tweet_count'].cumsum().to_list()
        consensus_class = []

        # Sum is faster and achieves the same purpose as mean in the argmax context
        for i in tqdm(range(len(df_pred_user)), desc=f'calculating consensus seed{SEED} {model_name}'):
            consensus_class.append(logits[cum_sum[i]: cum_sum[i+1]].sum(axis=0).argmax(axis=1)[0, 0])

        df_pred_user['pred_class_id'] = consensus_class
        df_pred_user = df_pred_user.join(class_mapper.add_prefix('pred_'), on='pred_class_id')
        df_pred_user.to_csv(f'seeded-{SEED}/results/authlist/authlist_user_preds_{model_name}.csv', index=False)


if __name__ == '__main__':

    main(SEED=0)