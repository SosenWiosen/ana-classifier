from sklearn.metrics import classification_report


def save_classification_report(y_true, y_pred, file_path):
    report = classification_report(y_true, y_pred)
    with open(file_path, 'w') as f:
        f.write(report)
