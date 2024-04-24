from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np


def accuracy_precision_recall1(score_list, label_list):
    precision, recall, threshold = precision_recall_curve(label_list, score_list)
    score = abs((precision / recall) - 1)
    idx = np.argmin(score)
    precision = precision[idx]
    recall = recall[idx]
    threshold = threshold[idx]

    tp, fn, tn, fp = 0, 0, 0, 0
    for gt, score in zip(label_list, score_list):
        pred = 1 if score > threshold else 0
        if gt == 1 and pred == 1:
            tp += 1
        if gt == 0 and pred == 0:
            tn += 1
        if gt == 1 and pred == 0:
            fn += 1
        if gt == 0 and pred == 1:
            fp += 1

    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return accuracy * 100, precision * 100, recall * 100, threshold


def accuracy_precision_recall2(pred_list, label_list):
    tp, fn, tn, fp = 0, 0, 0, 0
    for gt, pred in zip(label_list, pred_list):
        if gt == 1 and pred == 1:
            tp += 1
        if gt == 0 and pred == 0:
            tn += 1
        if gt == 1 and pred == 0:
            fn += 1
        if gt == 0 and pred == 1:
            fp += 1

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = 0
    if tp + fp != 0:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy * 100, precision * 100, recall * 100

