import json

import numpy as np

from model_tester.graph.export_confusion_matrix import export_confusion_matrix
from model_tester.graph.history_plot import save_plots_separately, save_combined_plot
from model_tester.model.train_model import train_model
from model_tester.model.train_model_k_fold import train_model_k_fold
import os
from datetime import datetime

from model_tester.save_report import save_classification_report
from model_tester.save_training_times import save_training_times


def test_model_k_fold(save_path="", dst_path="", model_name="", data_augmentation="", top="avgpool",
                      top_dropout_rate=0.2,
                      max_epochs=20,
                      optimizer="adam", early_stopping=None, metrics=None, finetune=False, finetune_layers=20,
                      finetune_optimizer="adam",
                      finetune_early_stopping=None, finetune_max_epochs=10, attempt_name=""):
    if attempt_name == "":
        attempt_name = model_name

    # Format the current datetime and create a path
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_directory = os.path.join(str(save_path), str(attempt_name) + "-" + str(current_time))
    finetune_plots_directory = os.path.join(save_directory, "train_finetune_results")
    plots_directory = os.path.join(save_directory, "train_results")
    test_results_directory = os.path.join(save_directory, "test_results")

    # Check if the directory exists, create if doesn't
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if not os.path.exists(finetune_plots_directory) and finetune:
        os.makedirs(finetune_plots_directory)
    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)
    if not os.path.exists(test_results_directory):
        os.makedirs(test_results_directory)

    test_path = os.path.join(dst_path, "test")
    train_path = os.path.join(dst_path, "train")
    k = 5
    # use train_model_k_fold() instead of train_model()
    # get fold_val_accuracies, fold_val_f1_scores, fold_test_accuracies, fold_test_f1_scores, (fold_val_y_trues, fold_val_y_preds), (fold_test_y_trues, fold_test_y_preds), fold_histories, fold_train_times, finetune_fold_val_accuracies, finetune_fold_val_f1_scores, finetune_fold_test_accuracies, finetune_fold_test_f1_scores, (finetune_fold_val_y_trues, finetune_fold_val_y_preds), (finetune_fold_test_y_trues, finetune_fold_test_y_preds), finetune_fold_histories, finetune_fold_train_times
    (
        class_names,  # Class names
        fold_val_accuracies,  # List of validation accuracies for each fold
        fold_val_f1_scores,  # List of validation F1-scores for each fold
        fold_test_accuracies,  # List of test accuracies for each fold
        fold_test_f1_scores,  # List of test F1-scores for each fold

        # True and predicted labels for validation and test sets
        original_tuple,  # (fold_val_y_trues, fold_val_y_preds) for validation set (all folds)
        test_original_tuple,  # (fold_test_y_trues, fold_test_y_preds) for test set (all folds)

        # Training and runtime information
        fold_histories,  # Training histories for each fold
        fold_train_times,  # Training times for each fold

        # Fine-tuning outputs
        finetune_fold_val_accuracies,  # Fine-tuning validation accuracies (per fold)
        finetune_fold_val_f1_scores,  # Fine-tuning validation F1-scores (per fold)
        finetune_fold_test_accuracies,  # Fine-tuning test accuracies (per fold)
        finetune_fold_test_f1_scores,  # Fine-tuning test F1-scores (per fold)

        # True and predicted labels for fine-tuning
        finetune_tuple,  # (finetune_fold_val_y_trues, finetune_fold_val_y_preds) for validation set
        test_finetune_tuple,  # (finetune_fold_test_y_trues, finetune_fold_test_y_preds) for test set

        # Fine-tuning histories and runtime
        finetune_fold_histories,  # Fine-tuning training histories for each fold
        finetune_fold_train_times  # Fine-tuning train times for each fold
    ) = train_model_k_fold(
        train_path,  # Training directory
        test_path,  # Test directory
        model_name,  # Model name (e.g., "resnet50")
        data_augmentation,  # Data augmentation layers
        top,  # Custom top layers
        top_dropout_rate,  # Dropout rate for top layers
        optimizer,  # Optimizer for training
        early_stopping,  # Early stopping callback
        metrics,  # List of metrics for evaluation
        finetune,  # Whether to fine-tune the model
        finetune_layers,  # Number of layers to fine-tune
        finetune_optimizer,  # Fine-tuning optimizer
        finetune_early_stopping,  # Early stopping for fine-tuning
        save_directory,
        k=k,
        max_epochs=max_epochs,  # Max epochs for training
        finetune_max_epochs=finetune_max_epochs  # Max epochs for fine-tuning
    )
    for i in range(k):
        y_true, y_pred = original_tuple[i]
        test_y_true, test_y_pred = test_original_tuple[i]
        history = fold_histories[i]
        train_times = fold_train_times[i]
        export_confusion_matrix(y_true, y_pred, class_names, plots_directory, filename=f'cfm_fold_{i}')
        save_plots_separately(history, plots_directory, filename=f'history_fold_{i}')
        save_combined_plot(history, plots_directory, filename=f'combined_fold_{i}')
        save_classification_report(y_true, y_pred,
                                   os.path.join(plots_directory, f"classification_report_fold_{i}.json"), class_names)
        save_training_times(train_times, os.path.join(plots_directory, f"training_times_fold_{i}.json"))
        save_classification_report(test_y_true, test_y_pred,
                                   os.path.join(test_results_directory, f"classification_report_fold_{i}.json"),
                                   class_names)
        export_confusion_matrix(test_y_true, test_y_pred, class_names, test_results_directory,
                                filename=f'cfm_test_fold_{i}')
        # fold_test_accuracies contains an array of accuracies from each fold, eg fold_test_accuracies[0] is the accuracy of fold 0. save the accuracy and f1 scores and their means
        with open(os.path.join(test_results_directory, f"test_scores_fold.json"), 'w') as f:
            json.dump({
                "test_accuracies": fold_test_accuracies[i],
                "test_f1_scores": fold_test_f1_scores[i],
                "test_avg_accuracy": np.mean(fold_test_accuracies[i]),
                "test_avg_f1_score": np.mean(fold_test_f1_scores[i])
            }, f, indent=4)

        if finetune:
            finetune_y_true, finetune_y_pred = finetune_tuple[i]
            test_finetune_y_true, test_finetune_y_pred = test_finetune_tuple[i]
            finetune_history = finetune_fold_histories[i]
            finetune_train_times = finetune_fold_train_times[i]
            export_confusion_matrix(finetune_y_true, finetune_y_pred, class_names, finetune_plots_directory,
                                    filename=f'cfm_finetune_fold_{i}')
            save_plots_separately(finetune_history, finetune_plots_directory, filename=f'history_finetune_fold_{i}')
            save_combined_plot(finetune_history, finetune_plots_directory, filename=f'combined_finetune_fold_{i}')
            save_classification_report(finetune_y_true, finetune_y_pred,
                                       os.path.join(finetune_plots_directory, f"classification_report_fold_{i}.json"),
                                       class_names)
            save_training_times(finetune_train_times,
                                os.path.join(finetune_plots_directory, f"training_times_fold_{i}.json"))
            save_classification_report(test_finetune_y_true, test_finetune_y_pred,
                                       os.path.join(test_results_directory,
                                                    f"classification_report_finetune_fold_{i}.json"),
                                       class_names)
            export_confusion_matrix(test_finetune_y_true, test_finetune_y_pred, class_names, test_results_directory,
                                    filename=f'cfm_test_finetune_fold_{i}')
            # save the f1 and accuracy scores arrays and their averages
            with open(os.path.join(test_results_directory, f"test_finetune_scores.json"), 'w') as f:
                json.dump({
                    "test_accuracies": finetune_fold_test_accuracies[i],
                    "test_f1_scores": finetune_fold_test_f1_scores[i],
                    "test_avg_accuracy": np.mean(finetune_fold_test_accuracies[i]),
                    "test_avg_f1_score": np.mean(finetune_fold_test_f1_scores[i])
                }, f, indent=4)

    # if finetune:
    #     finetune_y_true, finetune_y_pred = finetune_tuple
    #     test_finetune_y_true, test_finetune_y_pred = test_finetune_tuple
    #     export_confusion_matrix(finetune_y_true, finetune_y_pred, class_names, finetune_plots_directory)
    #     save_plots_separately(finetune_history, finetune_plots_directory)
    #     save_combined_plot(finetune_history, finetune_plots_directory)
    #     save_classification_report(finetune_y_true, finetune_y_pred,
    #                                os.path.join(finetune_plots_directory, "classification_report.json"), class_names)
    #     save_training_times(train_times, os.path.join(finetune_plots_directory, "training_times.json"))
    #     save_classification_report(test_finetune_y_true, test_finetune_y_pred,
    #                                os.path.join(test_results_directory, "classification_report_finetune.json"),
    #                                class_names)
    #     export_confusion_matrix(test_finetune_y_true, test_finetune_y_pred, class_names, test_results_directory,
    #                             filename='cfm_test_finetune')
