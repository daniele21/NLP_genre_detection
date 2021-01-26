from scripts.pipeline.dataset_pipeline import generate_training_dataset

def test_generate_training_dataset():
    params = {'train': True,
              'split_size': 0.8,
              'shuffle': True,
              'seed': 2021,
              'preload': 'resources/data/prep_data_v1.csv'}

    x_train, x_test, y_train, y_test = generate_training_dataset(params)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

if __name__ == '__main__':


    test_generate_training_dataset()

