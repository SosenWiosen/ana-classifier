from model_tester.graph.export_confusion_matrix import export_confusion_matrix
from model_tester.graph.history_plot import save_plots_separately, save_combined_plot
from model_tester.model.train_model import train_model
import os
from datetime import datetime

from model_tester.save_report import save_classification_report


def test_model(save_path, dst_path, model_name, data_augmentation, head="avgpool", top_dropout_rate=0.2,
               optimizer="adam", early_stopping=None, metrics=None, finetune=False, finetune_layers=20,
               finetune_optimizer="adam",
               finetune_early_stopping=None, attempt_name=""):
    if attempt_name == "":
        attempt_name = model_name

    # Format the current datetime and create a path
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_directory = os.path.join(str(save_path), str(current_time) + "-" + str(attempt_name))
    finetune_plots_directory = os.path.join(save_directory, "finetune_plots")
    plots_directory = os.path.join(save_directory, "plots")

    # Check if the directory exists, create if doesn't
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if not os.path.exists(finetune_plots_directory) and finetune:
        os.makedirs(finetune_plots_directory)
    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)

    model, class_names, history, original_tuple, finetune_history, finetune_tuple, = train_model(
        dst_path, model_name, data_augmentation, head, top_dropout_rate, optimizer, early_stopping, metrics, finetune,
        finetune_layers, finetune_optimizer, finetune_early_stopping, save_directory)
    y_true, y_pred = original_tuple
    export_confusion_matrix(y_true, y_pred, class_names, plots_directory)
    save_plots_separately(history, plots_directory)
    save_combined_plot(history, plots_directory)
    save_classification_report(y_true, y_pred, os.path.join(plots_directory, "classification_report.txt"))

    if finetune:
        finetune_y_true, finetune_y_pred = finetune_tuple
        export_confusion_matrix(finetune_y_true, finetune_y_pred, class_names, finetune_plots_directory)
        save_plots_separately(finetune_history, finetune_plots_directory)
        save_combined_plot(finetune_history, finetune_plots_directory)
        save_classification_report(finetune_y_true, finetune_y_pred,
                                   os.path.join(finetune_plots_directory, "classification_report.txt"))
