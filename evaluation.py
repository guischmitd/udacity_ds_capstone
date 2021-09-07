from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def evaluate_predictions(y_test, y_pred, metrics=[accuracy_score, precision_score, recall_score, f1_score, roc_auc_score], target_names=['ham', 'spam'], cmap='coolwarm_r', report=True, plot=True, norm_confusion_matrix=False):
    results = {m.__name__: m(y_test, y_pred) for m in metrics}

    if plot:
        cnf_mat = confusion_matrix(y_test, y_pred)
        cnf_mat_norm = cnf_mat / cnf_mat.sum(axis=0)

        annot = np.empty(cnf_mat.shape, dtype='object')
        for i, j in product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
            annot[i, j] = f'{cnf_mat[i, j]}\n({cnf_mat_norm.round(3)[i, j]})'

        sns.heatmap(cnf_mat_norm if norm_confusion_matrix else cnf_mat, annot=annot, cmap=cmap, fmt='')
        
        plt.ylabel('True')
        plt.xlabel('Predicted')

    if report:
        print(classification_report(y_test, y_pred, target_names=target_names))

    return pd.Series(results)


def evaluate_thresholds(y_test, y_prob, threshold_list=np.linspace(0, 1, 50), target_class=1, metrics=[accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]):
    results = []
    for threshold in threshold_list:
        y_pred = (y_prob[:, target_class] > threshold) * 1.0
        results.append(evaluate_predictions(y_test, y_pred, metrics=metrics, report=False, plot=False))
    
    return pd.concat(results, axis=1).T.set_index(threshold_list)
