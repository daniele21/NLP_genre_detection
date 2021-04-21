from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from constants.config import ADAM_OPTIMIZER, BCE_LOSS


def get_optimizer(optim):

    if isinstance(optim, str):
        if optim == ADAM_OPTIMIZER:
            return Adam
        else:
            raise ValueError(f'Optimizer not found: {optim}')

    else:
        return optim


def get_loss(loss):

    if isinstance(loss, str):
        if loss == BCE_LOSS:
            return BinaryCrossentropy(from_logits=True)
        else:
            raise ValueError(f'Loss function not found: {loss}')

    else:
        return loss
