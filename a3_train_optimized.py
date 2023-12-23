import tensorflow as tf  # v2.5.0
import numpy as np
import logging, time
import argparse, os
from tqdm import tqdm
from sklearn.metrics import accuracy_score

SEED = 0

import random
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

from utils.dataset import TCD_Selector
import utils.model


def main(args):

    ## Debugging code
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    # tf.config.experimental.set_memory_growth(physical_devices[args.gpu], enable=True)    

    # Create relevant folders and loggers

    EXP_NAME = f'features_{args.feat}_lr_1e-{args.lr}_ndim_{args.ndim}_nlstm_{args.nlstm}'
    INITIAL_LR = 1.0 / (10 ** args.lr)
    EPOCHS, BATCH_SIZE = 5, 64

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # Don't need for HPC since only one GPU
    os.makedirs(f'seeded-{SEED}/logs', exist_ok=True)
    os.makedirs(f'seeded-{SEED}/checkpoints/{EXP_NAME}', exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f'seeded-{SEED}/logs/lstm_run_{EXP_NAME}.log'),
            logging.StreamHandler()
        ]
    )


    '''
    Train LSTM model. Further definitions of model layers are in utils/model.py
    '''

    # Initialize data loaders, model and tf objects

    dg_train = TCD_Selector[args.feat](json_filepath='data/prep_3_train_split_801010.json', batch_size=BATCH_SIZE, features=args.feat)
    dg_val = TCD_Selector[args.feat](json_filepath='data/prep_3_val_split_801010.json', batch_size=BATCH_SIZE, features=args.feat)

    NUM_LOOP_TRAIN, NUM_LOOP_VAL = len(dg_train), len(dg_val)
    TOTAL_STEPS = EPOCHS * NUM_LOOP_TRAIN

    model = utils.model.init_lstm(embed_dim=args.ndim, nlstm=args.nlstm)
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [3*NUM_LOOP_TRAIN, 4*NUM_LOOP_TRAIN], [INITIAL_LR, INITIAL_LR/10, INITIAL_LR/100]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


    '''
    The train_step calls the model and optimizer to perform a single step of training.
    The logits are the raw predictions of the model, and the loss_fn calculates the loss.
    Gradients are calculated and applied to the model weights using the optimizer.

    The test_step is similar, but uses the validation set instead, and without backpropagation of gradients.
    '''

    # Optimized functions for training and validation

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value, logits

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_loss = loss_fn(y, val_logits)
        return val_loss, val_logits


    # Model warmup and initialize loop variables

    logging.info('Warming up model ...')
    test_step(tf.zeros([BATCH_SIZE, 256], dtype=tf.int32), tf.zeros([BATCH_SIZE], dtype=tf.int32))
    logging.info(f'Model warmup complete, training started ...')
    logging.info(f'Settings    : num_loop_train={NUM_LOOP_TRAIN}, num_loop_val={NUM_LOOP_VAL}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, gpu={args.gpu}')
    logging.info(f'Parameters  : seed={SEED}, features={args.feat}, initial_lr=1e-{args.lr}, ndim={args.ndim}, nlstm={args.nlstm}')

    elapsed_time = 0
    train_time_start = time.time()
    total_loss, total_acc = 0, 0

    for step in range(1, TOTAL_STEPS+1):  ## Exactly 5 epochs for batch size 64, increase if batch size is lowered
        
        '''
        Get a batch of tokenized tweets and train the model.
        '''

        # Get a batch of training data and train
        x_batch_train, y_batch_train = dg_train.get_batch()
        loss_acc = train_step(x_batch_train, y_batch_train)
        total_loss += loss_acc[0].numpy()
        total_acc += accuracy_score(y_batch_train.numpy(), np.argmax(loss_acc[1].numpy(), axis=1))

        # Print training stats every 500 steps
        if step % 500 == 0:
            time_diff = time.time() - train_time_start
            elapsed_time += time_diff
            logging.info(f'split=train | steps={step} | samples={step*BATCH_SIZE} | lr={optimizer.lr(step=step):.7f} ' + \
                f'| loss={total_loss/500:.5f} | acc={total_acc/500:.4f} | time={time_diff:.1f} ' + \
                f'| elapsed={elapsed_time:.0f} | remaining={(TOTAL_STEPS+1-step)/step*elapsed_time:.0f}')
            train_time_start = time.time()
            total_loss, total_acc = 0, 0

        # Validation step every 10k steps
        if step % 10000 == 0:
            total_val_loss, total_val_acc = 0, 0
            for _ in tqdm(range(NUM_LOOP_VAL), desc=f'val set inference @ step {step}'):
                x_batch_val, y_batch_val = dg_val.get_batch()
                val_loss_acc = test_step(x_batch_val, y_batch_val)
                total_val_loss += val_loss_acc[0].numpy()
                total_val_acc += accuracy_score(y_batch_val.numpy(), np.argmax(val_loss_acc[1].numpy(), axis=1))
            logging.info(f'split=val | steps={step} ' + \
                f'| val_loss={total_val_loss/NUM_LOOP_VAL:.5f} | val_acc={total_val_acc/NUM_LOOP_VAL:.4f}')
            train_time_start = time.time()
            # tf.keras.models.save_model(model, f'seeded-{SEED}/checkpoints/{EXP_NAME}/lstm-step-{step}.h5')

        # Additional debugging step
        if step == 50:
            logging.info('[DEBUG] Model training successful for 50 steps ...')
    
    # Final validation step - we use this checkpoint as the final model as it is usually the best
    total_val_loss, total_val_acc = 0, 0
    for _ in tqdm(range(NUM_LOOP_VAL), desc=f'val set inference @ step final {TOTAL_STEPS}'):
        x_batch_val, y_batch_val = dg_val.get_batch()
        val_loss_acc = test_step(x_batch_val, y_batch_val)
        total_val_loss += val_loss_acc[0].numpy()
        total_val_acc += accuracy_score(y_batch_val.numpy(), np.argmax(val_loss_acc[1].numpy(), axis=1))
    logging.info(f'split=val | steps={TOTAL_STEPS} ' + \
        f'| val_loss={total_val_loss/NUM_LOOP_VAL:.5f} | val_acc={total_val_acc/NUM_LOOP_VAL:.4f}')
    tf.keras.models.save_model(model, f'seeded-{SEED}/checkpoints/{EXP_NAME}/lstm-step-final-{TOTAL_STEPS}.h5')


if __name__ == '__main__':

    '''
    Hyperparameter search on the learning rates, embedding dimensions, number of LSTM layers (training part).

    Usage (use argument flags to specify the hyperparameters):
        python a3_train_optimized.py -gpu 0 -feat all -lr 3 -ndim 100 -nlstm 2
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=0, help='gpu id')
    parser.add_argument('-feat', type=str, default='all', choices=TCD_Selector.keys(), help='tweet features')
    parser.add_argument('-lr', type=int, default=3, help='initial learning rate')
    parser.add_argument('-ndim', type=int, default=100, help='word embedding dimension')
    parser.add_argument('-nlstm', type=int, default=2, help='number of lstm layers')
    args = parser.parse_args()

    main(args)
