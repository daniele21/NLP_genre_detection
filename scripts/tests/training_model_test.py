from scripts.training.model_training import train_model
import keras
import tensorflow as tf
from scripts.visualization.plots import plot_loss


def training_test():
    params = {}
    params['data'] = {'train': True,
                      'split_size': 0.8,
                      'shuffle': True,
                      'seed': 2021,
                      'preload': 'resources/data/prep_data_v1.csv'}

    params['network'] = {'n_word_tokens':None,
                         'n_classes': None,
                         'word_emb_size':50,
                         'weights': None,
                         'trainable': True,
                         'lstm_units': 30,
                         'dropout_rate': 0.3,
                         'optimizer': keras.optimizers.Adam,
                         'lr': 1e-2,
                         'loss': keras.losses.sparse_categorical_crossentropy,
                         }

    params['training'] = {'batch_size': 32,
                          'epochs': 1}

    model = train_model(params)

    plot_loss(model.history.history['loss'], model.history.history['val_loss'])


if __name__ == '__main__':

    tf.debugging.set_log_device_placement(True)

    training_test()