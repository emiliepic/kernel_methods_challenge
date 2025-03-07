from sklearn.metrics import f1_score

def f1_score_m1_1(y_true, y_pred):
    y_true_0_1 = (y_true == 1).astype(int)
    y_pred_0_1 = (y_pred == 1).astype(int)
    return f1_score(y_true_0_1, y_pred_0_1)