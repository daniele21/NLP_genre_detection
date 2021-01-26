from matplotlib import pyplot as plt

def plot_loss(train_loss, valid_loss):

    plt.plot(train_loss, label='Training loss')
    plt.plot(valid_loss, label='Valid loss')

    plt.legend()
    plt.grid()
    plt.show()