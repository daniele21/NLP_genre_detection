from keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, roc_auc_score, precision_score, recall_score

from core.file_manager.savings import save_json


def mean_top_k_accuracy_score(y_true, y_pred, k):
    acc = top_k_categorical_accuracy(y_true, y_pred, k).numpy()
    return acc.mean()

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def class_report(y_true, y_pred):
    report = multilabel_confusion_matrix(y_true, y_pred)
    return report

def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average='samples', multi_class='ovr')

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='samples')

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='samples')

def evaluation_report(y_true, y_pred, y_bin_pred, k=5, save_dir=None):

    report = {'acc_top_5': mean_top_k_accuracy_score(y_true, y_pred, k),
              'acc': accuracy(y_true, y_bin_pred),
              'prec': precision(y_true, y_bin_pred),
              'rec': recall(y_true, y_bin_pred),
              'auc': roc_auc(y_true, y_pred)}

    if save_dir is not None:
        filepath = f'{save_dir}evaluation_report.json'
        save_json(report, filepath=filepath)

    return report

