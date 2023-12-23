import numpy as np
from tqdm import tqdm
import scipy.sparse as sps
import os, argparse

import utils.model
from utils.dataset import TCD_Selector
from sklearn.metrics import accuracy_score, balanced_accuracy_score

SEED = 0


def tfarray2csr(array):
    return sps.csr_matrix((255*array.numpy()).astype(np.uint8))


def main(args, BATCH_SIZE=64):

    # Set GPU and create relevant folders
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.makedirs(f'seeded-{SEED}/results/{args.exp}', exist_ok=True)
    
    model = utils.model.load_pretrained(f'seeded-{SEED}/checkpoints/{args.exp}/lstm-step-{args.step}.h5')

    for split in args.split:

        '''
        Test on the dataset and save logits and predictions to file.
        '''

        # Test on the dataset
        dg = TCD_Selector[args.feat](json_filepath=f'data/prep_3_{split}_split_801010.json',
                                     batch_size=BATCH_SIZE, features=args.feat)
        y_all, logits_all = [], []

        for _ in tqdm(range(len(dg)), desc=f'[seed]={SEED} {args.exp} | {split}'):
            x_batch, y_batch = dg.get_batch()
            logits_all.append(tfarray2csr(model(x_batch, training=False)))
            y_all.append(y_batch.numpy())

        y_true = np.concatenate(y_all, axis=0)[:dg.total_lines]
        y_logits = sps.vstack(logits_all)[:dg.total_lines]
        y_pred = np.squeeze(np.asarray(y_logits.argmax(axis=1)))

        sps.save_npz(f'seeded-{SEED}/results/{args.exp}/logits_dataset_{split}_{args.step}.npz', y_logits)
        np.savez(f'seeded-{SEED}/results/{args.exp}/y_dataset_{split}_{args.step}.npz', y_true=y_true, y_pred=y_pred)
        
        print(f'split={split}, acc1={accuracy_score(y_true, y_pred):.4f}, balacc={balanced_accuracy_score(y_true, y_pred):.4f}')


if __name__ == '__main__':

    '''
    Hyperparameter search on the learning rates, embedding dimensions, number of LSTM layers (evaluation on validation set part).

    Usage (use argument flags to specify the hyperparameters):
        python a4_lstm_test.py -gpu 0 -exp features_all_lr_1e-3_ndim_100_nlstm_2 -feat all -split val
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=0, help='gpu id')
    parser.add_argument('-exp', type=str, required=True, help='experiment name / directory')
    parser.add_argument('-feat', type=str, default='all', choices=TCD_Selector.keys(), help='tweet features')
    parser.add_argument('-split', type=str, choices=['train', 'val', 'test'], nargs='+', help='dataset split')
    parser.add_argument('-step', type=str, default='final-245475', help='checkpoint step')
    args = parser.parse_args()

    main(args)