import os

import numpy as np
import math
import gc
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


def train_model_k_fold(dst_path, model_name, data_augmentation_factory, k=4, head="avgpool", top_dropout_rate=0.2,
                       optimizer_factory=lambda: "adam", early_stopping_factory= lambda: [], metrics=None, finetune=False, finetune_layers=20,
                       finetune_optimizer_factory= lambda: "adam",
                       finetune_early_stopping_factory=lambda: [], model_save_path=None, max_epochs=20, finetune_max_epochs=10):
    shape = get_input_shape(model_name)

    img_height, img_width = shape[:2]  # Extract the first two elements
    batch_size = 32


    def copy_red_to_green_and_blue(image, label):
        red_channel = image[..., 0:1]  # Extract only the red channel, shape (H, W, 1)
        new_image = tf.concat([red_channel, red_channel, red_channel], axis=-1)
        return new_image, label
    def load_full_dataset(path):
        """Load the dataset from a directory using Keras' utility and repeat it for cyclical iterating."""
        dataset = tf.keras.utils.image_dataset_from_directory(
            path,
            labels="inferred",
            label_mode="categorical",
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=True,
            seed=123
        )
        return dataset

    def prepare_dataset(dataset):
        """Apply transformations like caching, batching, and prefetching."""
        dataset = dataset.map(copy_red_to_green_and_blue)  # Apply preprocessing
        dataset = dataset.cache()  # Cache for better performance
        # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch for overlap
        return dataset

    def split_train_val(train_dataset, train_size=0.8):
        """Split a train dataset into train and validation datasets (80%-20%)."""
        # Get size of the dataset
        dataset_size = sum(1 for _ in train_dataset)
        train_split_size = int(train_size * dataset_size)

        # Split: Take `train_split_size` for training and the rest for validation
        train_ds = train_dataset.take(train_split_size)  # Repeat for continuous iteration
        val_ds = train_dataset.skip(train_split_size)# Repeat for continuous iteration
        return train_ds, val_ds
    
    # Function to perform TensorFlow's dataset partitioning for train/val/test
    def get_k_fold_splits(dataset, dataset_size, folds=5):
        """Partition dataset into K folds."""
        indices = tf.range(dataset_size)
        indices = tf.random.shuffle(indices, seed=123)
        fold_size = math.ceil(dataset_size / folds)
        folds_datasets = []

        for i in range(folds):
            start = i * fold_size
            end = start + fold_size

            test_indices = indices[start:end]
            train_indices = tf.concat([indices[:start], indices[end:]], axis=0)

            train_ds = dataset.enumerate().filter(
                lambda idx, data: tf.reduce_any(tf.equal(tf.cast(idx, tf.int64), tf.cast(train_indices, tf.int64)))
            ).map(lambda idx, data: data)  # Convert back to original structure

            test_ds = dataset.enumerate().filter(
                lambda idx, data: tf.reduce_any(tf.equal(tf.cast(idx, tf.int64), tf.cast(test_indices, tf.int64)))
            ).map(lambda idx, data: data)  # Convert back to original structure

            folds_datasets.append((train_ds, test_ds))

        return folds_datasets
    def compute_class_weights(train_ds):
        """Compute class weights for the given training dataset."""
        # Aggregate all the labels in the training dataset
        all_labels = []
        for _, labels in train_ds.unbatch():
            # Extract labels (assumed to be one-hot encoded)
            all_labels.append(tf.argmax(labels).numpy())

        # Convert list of labels to numpy array
        label_indices = np.array(all_labels)

        # Compute class weights using sklearn's class_weight utility:
        class_weights_array = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(label_indices),
            y=label_indices
        )

        # Convert to dictionary format required by TensorFlow
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

        return class_weights
    raw_dataset = load_full_dataset(dst_path)  # Load the dataset
    class_names = raw_dataset.class_names      # Extract class names
    dataset = prepare_dataset(raw_dataset)     # Prepare the dataset (cache, map, prefetch, etc.)

    # Get dataset size
    dataset_size = sum(1 for _ in dataset.unbatch())  # For counting the total number of examples

    # Implement K-Fold Cross-Validation
    folds = get_k_fold_splits(dataset, dataset_size, folds=k)
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

    for fold_no, (train_val_ds, test_dataset) in enumerate(folds, start=1):
        print(f"---------------- Fold {fold_no} ----------------")

        # Create a list of all labels from the dataset
        all_labels = []
        for _, labels in dataset.unbatch():  # Unbatch the dataset to get individual labels
            all_labels.append(tf.argmax(labels).numpy())  # Assuming labels are one-hot encoded
        num_classes = len(np.unique(all_labels))  # Count unique classes
        print('Found num of classes.')
        train_ds, val_ds = split_train_val(train_val_ds)
        class_weights = compute_class_weights(train_ds)

        # Apply `.repeat()` after class weights are computed
        train_ds = train_ds  # Repeat for training
        val_ds = val_ds      # Repeat ONLY if desired for validation

        train_size = sum(1 for _ in train_ds.unbatch())  # Total training samples
        val_size = sum(1 for _ in val_ds.unbatch())      # Total validation samples

        steps_per_epoch = train_size // batch_size       # Calculate training steps
        validation_steps = val_size // batch_size        # Calculate validation steps




        if metrics is None:
            metrics = [f1]

        train_class_weights = dict(enumerate(class_weights))

        inputs = tf.keras.layers.Input(shape=shape)
        x = data_augmentation_factory()(inputs)

        preprocess_input = tf.keras.layers.Lambda(lambda x: get_preprocess_input(model_name)(x))
        x = preprocess_input(x)

        base_model = get_base_model(model_name, shape, x)
        base_model.trainable = False
        outputs = get_top(base_model.output, num_classes, head, top_dropout_rate)
        model = tf.keras.Model(inputs, outputs, name=model_name)
        print('before compile')
        model.compile(optimizer=optimizer_factory(), loss="categorical_crossentropy", metrics=["accuracy"] + metrics)

        train_time_callback = TimeHistory()
        checkpoint_path = os.path.join(model_save_path, f"{fold_no}_best_weights.keras") if model_save_path else f"{fold_no}_best_weights.keras"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            verbose=1,
            monitor="val_accuracy",  # Monitor validation accuracy
            mode="max",
        )

        history = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs,
                            steps_per_epoch=steps_per_epoch,  # Ensure finite training steps
                            validation_steps=validation_steps,  # Ensure finite validation steps
                            # class_weight=train_class_weights,
                            callbacks=[early_stopping_factory(), train_time_callback, model_checkpoint_callback])
        model.load_weights(checkpoint_path)
        # Validation Evaluation
        predictions = []
        y_true = []

        for images, labels in val_ds:
            preds = model.predict(images)
            predictions.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(labels, axis=1))  # Convert one-hot to integer labels

        y_pred = np.array(predictions)
        y_true = np.array(y_true)

        # Save the model for the current fold if needed
        if model_save_path:
            model_file_path = os.path.join(model_save_path, f"model_fold_{fold_no}.keras")
            model.save(model_file_path)
            export_path = os.path.join(model_save_path, f"model_fold_{fold_no}exported_model")
            model.export(export_path, verbose=True)  # Export lightweight TF SavedModel
            print(f"Model exported for inference to: {export_path}")

        # Test Evaluation
        test_predictions = []
        test_y_true = []

        for images, labels in test_dataset:
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
            optimizer=finetune_optimizer_factory(),
            loss="categorical_crossentropy",
            metrics=['accuracy'] + metrics
        )

        finetune_time_callback = TimeHistory()
        finetune_checkpoint_path = os.path.join(model_save_path, f"{fold_no}best_weights_finetune.keras") if model_save_path else f"{fold_no}best_weights_finetune.keras"
        finetune_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=finetune_checkpoint_path,
            save_best_only=True,
            verbose=1,
            monitor="val_accuracy",  # Monitor validation accuracy
            mode="max",
        )

        finetune_history = model.fit(train_ds, validation_data=val_ds,
                                     epochs=finetune_max_epochs,
                                     steps_per_epoch=steps_per_epoch,  # Ensure finite training steps
                                     validation_steps=validation_steps,  # Ensure finite validation steps
                                    #  class_weight=train_class_weights,
                                     callbacks=[finetune_early_stopping_factory(), finetune_time_callback, finetune_model_checkpoint_callback])
        model.load_weights(finetune_checkpoint_path)
        if model_save_path:
            model_file_path = os.path.join(model_save_path, f"model_fold_{fold_no}_finetune.keras")
            model.save(model_file_path)
            export_path = os.path.join(model_save_path, f"model_fold_{fold_no}exported_model")
            model.export(export_path, verbose=True)  # Export lightweight TF SavedModel
            print(f"Model exported for inference to: {export_path}")
        finetune_predictions = []
        finetune_y_true = []

        for images, labels in val_ds:
            preds = model.predict(images)
            finetune_predictions.extend(np.argmax(preds, axis=1))
            finetune_y_true.extend(np.argmax(labels, axis=1))  # Convert one-hot to integer labels

        y_pred = np.array(finetune_predictions)
        y_true = np.array(finetune_y_true)

        test_finetune_predictions = []
        test_finetune_y_true = []
        for images, labels in test_dataset:
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
        gc.collect()
        fold_no=fold_no+1

    if finetune:
        return class_names, fold_val_accuracies, fold_val_f1_scores, fold_test_accuracies, fold_test_f1_scores, (fold_val_y_trues, fold_val_y_preds), (fold_test_y_trues, fold_test_y_preds), fold_histories, fold_train_times, finetune_fold_val_accuracies, finetune_fold_val_f1_scores, finetune_fold_test_accuracies, finetune_fold_test_f1_scores, (finetune_fold_val_y_trues, finetune_fold_val_y_preds), (finetune_fold_test_y_trues, finetune_fold_test_y_preds), finetune_fold_histories, finetune_fold_train_times
    return class_names,  fold_val_accuracies, fold_val_f1_scores, fold_test_accuracies, fold_test_f1_scores, (fold_val_y_trues, fold_val_y_preds), (fold_test_y_trues, fold_test_y_preds), fold_histories, fold_train_times, None, None, None, None, None, None, None, None
