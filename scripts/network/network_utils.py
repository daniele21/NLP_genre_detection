from keras.optimizers import Adam
from keras.losses import binary_crossentropy

from constants.config import ADAM_OPTIMIZER, BCE_LOSS


def get_optimizer_from_name(optim_name):

    if(optim_name == ADAM_OPTIMIZER):
        return Adam

    else:
        raise ValueError(f'Optimizer not found: {optim_name}')

def get_loss_from_name(loss_name):

    if(loss_name == BCE_LOSS):
        return binary_crossentropy

    else:
        raise ValueError(f'Loss function not found: {loss_name}')