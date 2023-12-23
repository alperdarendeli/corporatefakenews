import tensorflow as tf
import numpy as np
from tqdm import tqdm
import scipy.sparse as sps
import os, argparse

from utils.dataset import TCD_Selector
import utils.model

def tfarray2csr(array):
    return sps.csr_matrix((255*array.numpy()).astype(np.uint8))


'''
Use the trained model parameters to predict the user locations in the prediction dataset.
'''

# Inference on the authlist dataset, similar to 04a_lstm_test.py except there are no ground truth labels, so indices are used instead

def batch_infer(args, BATCH_SIZE=128):

    json_filepath=f'data/authlist_to_infer_cleaned.json'
    os.makedirs(f'seeded-{args.SEED}/results/authlist', exist_ok=True)
    dg = TCD_Selector['all'](json_filepath=json_filepath, eval=True, batch_size=BATCH_SIZE, features='all')
    model = utils.model.load_pretrained(f'seeded-{args.SEED}/checkpoints/features_all_lr_1e-3_ndim_{args.ndim}_nlstm_{args.nlstm}/lstm-step-final-245475.h5')
    logits_all = []

    @tf.function
    def infer(x):
        return model(x, training=False)

    for _ in tqdm(range(len(dg)), desc=f'ndim={args.ndim}, nlstm={args.nlstm}'):
        x_batch, ind_batch = dg.get_batch()
        logits_all.append(tfarray2csr(infer(x_batch)))

    y_logits = sps.vstack(logits_all)[:dg.total_lines]
    sps.save_npz(f'seeded-{args.SEED}/results/authlist/logits_authlist_ndim_{args.ndim}_nlstm_{args.nlstm}.npz', y_logits)


if __name__ == '__main__':

    '''Run inference on the authlist dataset
    
    Usage (use argument flags to specify the hyperparameters):
    
        python b2_lstm_infer.py -SEED 0 -ndim 200 -nlstm 2'''

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('-SEED', type=int, choices=[0, 1, 1234, 123], help='seed')
    parser.add_argument('-ndim', type=int, default=100, help='word embedding dimension')
    parser.add_argument('-nlstm', type=int, default=2, help='number of lstm layers')
    args = parser.parse_args()

    batch_infer(args)