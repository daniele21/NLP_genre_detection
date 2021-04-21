from argparse import ArgumentParser
import logging

from scripts.pipeline.dataset_pipeline import generate_training_dataset
from scripts.pipeline.training_pipeline import training_pipeline
from scripts.training.model_training import train_model

logger = logging.getLogger(__name__)


def main(args):

    params = {}
    params['data'] = {'train': True,
                      'split_size': args.split_size,
                      'shuffle': args.shuffle,
                      'seed': args.seed,
                      'preload': args.preload,
                      'data_path': args.data_path,
                      'tokenizer': args.tokenizer}

    params['network'] = {'word_emb_size': args.emb_size,
                         'weights': None,
                         'trainable': True,
                         'lstm_units': args.lstm_units,
                         'dropout_rate': args.dropout,
                         # 'optimizer': tf.keras.optimizers.Adam,
                         'lr': args.lr,
                         # 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                         }

    params['training'] = {'batch_size': args.batch_size,
                          'epochs': args.epochs}

    dataset = args.dataset
    train = args.train

    if not dataset and train:
        model = training_pipeline(params)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='Training Pipeline Mode')
    parser.add_argument('-d', '--dataset', action='store_true', help='Dataset Pipeline Mode')

    parser.add_argument('--split_size', type=float, default=0.8, help='Split size for dataset')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset')
    parser.add_argument('--seed', type=int, default=2021, help='Seed')
    parser.add_argument('--preload', type=str, default='resources/data/prep_data_v2.csv', help='Preload for dataset')
    parser.add_argument('--data_path', type=str, default='resources/train.csv', help='Dataset to load')
    parser.add_argument('--tokenizer', type=str, default='custom', help='Tokenizer: custom or keras')

    parser.add_argument('--emb_size', type=int, default=50, help='Word embedding size')
    parser.add_argument('--lstm_units', type=int, default=30, help='LSTM units')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')

    parsed_args = parser.parse_args()

    main(parsed_args)