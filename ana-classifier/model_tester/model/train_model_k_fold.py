import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.utils import class_weight

from model_tester.model.get_should_train_layer import get_should_train_layer
from model_tester.metrics.f1 import f1
from model_tester.model.get_base_model import get_base_model
from model_tester.metrics.time_history import TimeHistory
import tensorflow as tf

from model_tester.model.get_input_shape import get_input_shape
from model_tester.model.get_preprocess_input import get_preprocess_input
from model_tester.model.get_top import get_top


def train_model_k_fold(dst_path, model_name, data_augmentation, k=5, head="avgpool", top_dropout_rate=0.2,
                       optimizer="adam", early_stopping=None, metrics=None, finetune=False, finetune_layers=20,
                       finetune_optimizer="adam",
                       finetune_early_stopping=None, model_save_path=None, max_epochs=20, finetune_max_epochs=10):
    test_split_ratio = 0.2
    if finetune_early_stopping is None:
        finetune_early_stopping = []
    if early_stopping is None:
        early_stopping = []
    shape = get_input_shape(model_name)

    img_height, img_width = shape[:2]  # Extract the first two elements
    batch_size = 32

    def load_full_dataset(path):
        dataset = tf.keras.utils.image_dataset_from_directory(
            path,
            labels="inferred",
            label_mode="categorical",
            image_size=(img_height, img_width),
            shuffle=True,
            seed=123
        )
        return dataset

    dataset = load_full_dataset(dst_path)
    class_names = dataset.class_names
    num_classes = len(class_names)

    def copy_red_to_green_and_blue(image, label):
        red_channel = image[..., 0:1]  # Extract only the red channel, shape (H, W, 1)
        new_image = tf.concat([red_channel, red_channel, red_channel], axis=-1)
        return new_image, label

    dataset = dataset.map(copy_red_to_green_and_blue)
    # Splitting dataset into train, val, test
    all_images = []
    all_labels = []
    for img, lbl in dataset.unbatch():
        all_images.append(img.numpy())
        all_labels.append(lbl.numpy())

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    # def preprocess_dataset(ds):
    #     ds = ds.map(copy_red_to_green_and_blue)
    #     # ds = ds.prefetch(buffer_size=AUTOTUNE)
    #     return ds

    # def prepare_dataset(images, labels):
    #     ds = tf.data.Dataset.from_tensor_slices((images, labels))
    #     ds = ds.batch(batch_size)  # Batch the dataset
    #     ds = preprocess_dataset(ds)  # Apply preprocessing (e.g., red channel adjustment)
    #     return ds

    # Implement K-Fold Cross-Validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_no = 1
    fold_test_accuracies = []
    fold_test_f1_scores = []
    fold_val_accuracies = []
    fold_val_f1_scores = []
    fold_test_y_trues = []
    fold_test_y_preds = []
    fold_val_y_trues = []
    fold_val_y_preds = []
    fold_histories = []
    fold_train_times = []

    finetune_fold_histories = []
    finetune_fold_train_times = []
    finetune_fold_test_accuracies = []
    finetune_fold_test_f1_scores = []
    finetune_fold_test_y_trues = []
    finetune_fold_test_y_preds = []
    finetune_fold_val_accuracies = []
    finetune_fold_val_f1_scores = []
    finetune_fold_val_y_trues = []
    finetune_fold_val_y_preds = []

    for train_val_indices, test_indices in kfold.split(all_images):
        print(f"---------------- Fold {fold_no} ----------------")

        # Split into train+val and test
        train_val_images, test_images = all_images[train_val_indices], all_images[test_indices]
        train_val_labels, test_labels = all_labels[train_val_indices], all_labels[test_indices]

        # Further split train+val into train and validation (80%-20%)
        val_split_index = int(0.8 * len(train_val_images))
        train_images, val_images = train_val_images[:val_split_index], train_val_images[val_split_index:]
        train_labels, val_labels = train_val_labels[:val_split_index], train_val_labels[val_split_index:]

        # Compute class weights
        label_indices = np.argmax(train_labels, axis=1)
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(label_indices),
            y=label_indices
        )

        if metrics is None:
            metrics = [f1]

        train_class_weights = dict(enumerate(class_weights))

        inputs = tf.keras.layers.Input(shape=shape)
        x = data_augmentation(inputs)

        preprocess_input = tf.keras.layers.Lambda(lambda x: get_preprocess_input(model_name)(x))
        x = preprocess_input(x)

        base_model = get_base_model(model_name, shape, x)
        base_model.trainable = False
        outputs = get_top(base_model.output, num_classes, head, top_dropout_rate)
        model = tf.keras.Model(inputs, outputs, name=model_name)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"] + metrics)

        train_time_callback = TimeHistory()

        history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=max_epochs,
                            class_weight=train_class_weights,
                            callbacks=[early_stopping, train_time_callback])
        # Validation Evaluation
        predictions = []
        y_true = []

        for images, labels in tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size):
            preds = model.predict(images)
            predictions.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(labels, axis=1))  # Convert one-hot to integer labels

        y_pred = np.array(predictions)
        y_true = np.array(y_true)

        # Save the model for the current fold if needed
        if model_save_path:
            model_file_path = os.path.join(model_save_path, f"model_fold_{fold_no}.keras")
            model.save(model_file_path)

        # Test Evaluation
        test_predictions = []
        test_y_true = []

        for images, labels in tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size):
            preds = model.predict(images)
            test_predictions.extend(np.argmax(preds, axis=1))
            test_y_true.extend(np.argmax(labels, axis=1))  # Convert one-hot to integer labels

        test_y_pred = np.array(test_predictions)
        test_y_true = np.array(test_y_true)

        # Compute accuracy and F1 scores for validation and test
        val_accuracy = np.mean(y_pred == y_true)
        test_accuracy = np.mean(test_y_pred == test_y_true)

        from sklearn.metrics import f1_score
        val_f1 = f1_score(y_true, y_pred, average='weighted')
        test_f1 = f1_score(test_y_true, test_y_pred, average='weighted')

        # Store metrics for each fold
        fold_test_accuracies.append(test_accuracy)
        fold_test_f1_scores.append(test_f1)
        fold_histories.append(history)
        fold_train_times.append(train_time_callback.times)
        fold_val_accuracies.append(val_accuracy)
        fold_val_f1_scores.append(val_f1)
        fold_val_y_trues.append(y_true)
        fold_val_y_preds.append(y_pred)
        fold_test_y_trues.append(test_y_true)
        fold_test_y_preds.append(test_y_pred)
        if not finetune:
            continue

        should_train = get_should_train_layer(model_name)
        for layer in base_model.layers[-finetune_layers:]:
            if should_train(layer):
                layer.trainable = True

        model.compile(
            optimizer=finetune_optimizer,
            loss="categorical_crossentropy",
            metrics=['accuracy'] + metrics
        )

        finetune_time_callback = TimeHistory()

        finetune_history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels),
                                     epochs=finetune_max_epochs,
                                     class_weight=train_class_weights,
                                     callbacks=[finetune_early_stopping, finetune_time_callback])
        if model_save_path:
            model_file_path = os.path.join(model_save_path, f"model_fold_{fold_no}_finetune.keras")
            model.save(model_file_path)
        finetune_predictions = []
        finetune_y_true = []

        for images, labels in tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size):
            preds = model.predict(images)
            finetune_predictions.extend(np.argmax(preds, axis=1))
            finetune_y_true.extend(np.argmax(labels, axis=1))  # Convert one-hot to integer labels

        y_pred = np.array(finetune_predictions)
        y_true = np.array(finetune_y_true)

        test_finetune_predictions = []
        test_finetune_y_true = []
        for images, labels in tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size):
            preds = model.predict(images)
            test_finetune_predictions.extend(np.argmax(preds, axis=1))
            test_finetune_y_true.extend(np.argmax(labels, axis=1))  # Convert one-hot to integer labels

        test_finetune_y_pred = np.array(test_finetune_predictions)
        test_finetune_y_true = np.array(test_finetune_y_true)

        finetune_fold_histories.append(finetune_history)
        finetune_fold_train_times.append(finetune_time_callback.times)
        finetune_fold_test_accuracies.append(np.mean(test_finetune_y_pred == test_finetune_y_true))
        finetune_fold_test_f1_scores.append(f1_score(test_finetune_y_true, test_finetune_y_pred, average='weighted'))
        finetune_fold_val_accuracies.append(np.mean(y_pred == y_true))
        finetune_fold_val_f1_scores.append(f1_score(y_true, y_pred, average='weighted'))
        finetune_fold_val_y_trues.append(y_true)
        finetune_fold_val_y_preds.append(y_pred)
        finetune_fold_test_y_trues.append(test_finetune_y_true)
        finetune_fold_test_y_preds.append(test_finetune_y_pred)

    if finetune:
        return class_names, fold_val_accuracies, fold_val_f1_scores, fold_test_accuracies, fold_test_f1_scores, (fold_val_y_trues, fold_val_y_preds), (fold_test_y_trues, fold_test_y_preds), fold_histories, fold_train_times, finetune_fold_val_accuracies, finetune_fold_val_f1_scores, finetune_fold_test_accuracies, finetune_fold_test_f1_scores, (finetune_fold_val_y_trues, finetune_fold_val_y_preds), (finetune_fold_test_y_trues, finetune_fold_test_y_preds), finetune_fold_histories, finetune_fold_train_times
    return class_names,  fold_val_accuracies, fold_val_f1_scores, fold_test_accuracies, fold_test_f1_scores, (fold_val_y_trues, fold_val_y_preds), (fold_test_y_trues, fold_test_y_preds), fold_histories, fold_train_times, None, None, None, None, None, None, None, None
