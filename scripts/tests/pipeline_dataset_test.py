from scripts.pipeline.dataset_pipeline import generate_training_dataset


def test_generate_training_dataset():
    params = {'train': True,
              'split_size': 0.8,
              'shuffle': True,
              'seed': 2021,
              'preload': 'resources/data/prep_data_v2.csv',
              'data_path': 'resources/train.csv',
              'tokenizer': 'keras'}

    dataset, tokenizer = generate_training_dataset(params)

    print(dataset['train']['x'].shape)
    print(dataset['test']['x'].shape)
    print(dataset['train']['y'].shape)
    print(dataset['test']['y'].shape)


if __name__ == '__main__':

    test_generate_training_dataset()

