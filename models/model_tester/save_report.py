import json

from sklearn.metrics import classification_report


def save_classification_report(y_true, y_pred, file_path, target_names=None):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_json = json.dumps(report, indent=4)
    with open(file_path, 'w') as f:
        f.write(report_json)
