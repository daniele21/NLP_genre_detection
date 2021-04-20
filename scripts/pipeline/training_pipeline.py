from scripts.pipeline.dataset_pipeline import generate_training_dataset
from scripts.training.model_training import train_model
from scripts.visualization.plots import plot_loss


def training_pipeline(params, save_dir=None, callbacks=None):
    params['data']['train'] = True

    data_params = params['data']
    data_resources = generate_training_dataset(data_params, save_dir=save_dir)

    model = train_model(data_resources, params, callbacks=callbacks)

    plot_loss(model.history.history['loss'], model.history.history['val_loss'])

    return model

