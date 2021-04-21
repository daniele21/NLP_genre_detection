from datetime import datetime

from constants.paths import MODELS_DIR, GLOVE_PATH
from scripts.pipeline.training_pipeline import training_pipeline
from scripts.training.model_training import train_model
import keras
import tensorflow as tf

from scripts.utils.savings import save_params
from scripts.visualization.plots import plot_loss


def training_test():
    params = {}
    params['data'] = {'train': True,
                      'split_size': 0.8,
                      'shuffle': True,
                      'seed': 2021,
                      'preload': 'resources/data/prep_data_v2.csv',
                      'tokenizer': 'keras',
                      'data_path': 'resources/train.csv',
                      }

    params['network'] = {'word_emb_size':100,
                         'pretrained': True,
                         'emb_path': GLOVE_PATH,
                         'lstm_units': 128,
                         'dropout_rate': 0.3,
                         # 'optimizer': tf.keras.optimizers.Adam,
                         'lr': 1e-2,
                         # 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                         }

    params['training'] = {'batch_size': 64,
                          'epochs': 15}

    timestamp = str(datetime.now()).replace(' ', '_')
    model_name = f'model_v0_ckp_{timestamp}'
    model_dir = f'{MODELS_DIR}{model_name}/'
    filepaths = save_params(params, save_dir=model_dir)
    model_path = f'{model_dir}{model_name}.hdf5'

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1),
                 keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False,
                                                 monitor='val_acc', save_best_only=True,
                                                 mode='min', verbose=0)]

    model = training_pipeline(params, save_dir=model_dir, callbacks=callbacks)

    # plot_loss(model.history.history['loss'], model.history.history['val_loss'])


if __name__ == '__main__':

    # tf.debugging.set_log_device_placement(True)

    training_test()