import os

import keras_tuner
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

from model_tester.model.get_should_train_layer import get_should_train_layer
from model_tester.metrics.f1 import f1
from model_tester.model.get_base_model import get_base_model
from model_tester.metrics.time_history import TimeHistory
import tensorflow as tf

from model_tester.model.get_input_shape import get_input_shape
from model_tester.model.get_preprocess_input import get_preprocess_input
from model_tester.model.get_top import get_top


def hypertune_model(train_path, test_path, model_name, data_augmentation, working_dir, projectname='hypertuning',
                    early_stopping=None, metrics=None, finetune=False,
                    finetune_early_stopping=None, model_save_path=None, max_epochs=20, finetune_max_epochs=10):
    if finetune_early_stopping is None:
        finetune_early_stopping = []
    if early_stopping is None:
        early_stopping = []
    shape = get_input_shape(model_name)
    img_height, img_width = shape[:2]  # Extract the first two elements
    batch_size = 32

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=train_path,
        labels='inferred',
        subset="both",
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        validation_split=0.15,
        seed=123,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=test_path,
        labels='inferred',
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    def copy_red_to_green_and_blue(image, label):
        red_channel = image[..., 0:1]  # Extract only the red channel, shape (H, W, 1)
        new_image = tf.concat([red_channel, red_channel, red_channel], axis=-1)
        return new_image, label

    train_ds = train_ds.map(copy_red_to_green_and_blue)
    val_ds = val_ds.map(copy_red_to_green_and_blue)
    test_ds = test_ds.map(copy_red_to_green_and_blue)

    if metrics is None:
        metrics = [f1]
    labels = np.concatenate([y for x, y in train_ds], axis=0)
    label_indices = np.argmax(labels, axis=1)

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(label_indices),
        y=label_indices
    )

    train_class_weights = dict(enumerate(class_weights))

    def build_model(hp):
        inputs = tf.keras.layers.Input(shape=shape)
        x = data_augmentation(inputs)

        preprocess_input = tf.keras.layers.Lambda(lambda x: get_preprocess_input(model_name)(x))
        x = preprocess_input(x)

        base_model = get_base_model(model_name, shape, x)
        base_model.trainable = False
        hp_pooling = hp.Choice('pooling', values=['avg', 'max', 'flatten'])
        if (hp_pooling == 'avg'):
            x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        elif (hp_pooling == 'max'):
            x = layers.GlobalMaxPooling2D(name="max_pool")(base_model.output)
        elif (hp_pooling == 'flatten'):
            x= layers.Flatten()(base_model.output)
        hp_batch_norm = hp.Choice('batch_norm', values=[True, False])
        if hp_batch_norm:
            x = layers.BatchNormalization()(x)
        hp_neurons = hp.Int('neurons', min_value=512, max_value=4096, step=256)
        hp_dropout = hp.Float('dropout', min_value=0.01, max_value=0.4, step=0.02)
        x = layers.Dense(hp_neurons, activation='relu')(x)
        x = layers.Dropout(hp_dropout, name="top_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
        model = tf.keras.Model(inputs, outputs, name=model_name)
        hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 5e-6])
        model.compile(optimizer=Adam(learning_rate=hp_lr), loss="categorical_crossentropy",
                      metrics=["accuracy"] + metrics)
        return model

    train_time_callback = TimeHistory()

    tuner = keras_tuner.Hyperband(
        build_model,
        objective=keras_tuner.Objective("val_f1", direction="max"),
        max_epochs=100,
        factor=3,
        directory=working_dir,
        project_name=projectname,
    )
    # history = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, class_weight=train_class_weights,
    #                     callbacks=[early_stopping, train_time_callback])
    tuner.search(train_ds, validation_data=val_ds, epochs=max_epochs, class_weight=train_class_weights)
    summary = tuner.results_summary()
    predictions = []
    y_true = []
    best_hps = tuner.get_best_hyperparameters(5)
    # Build the model with the best hp.
    model = build_model(best_hps[0])
    history = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, class_weight=train_class_weights,
                        callbacks=[early_stopping, train_time_callback])

    for images, labels in val_ds:
        preds = model.predict(images)
        predictions.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))  # Convert one-hot to integer labels

    y_pred = np.array(predictions)
    y_true = np.array(y_true)
    if model_save_path:
        model_file_path = os.path.join(model_save_path, "model.keras")
        model.save(model_file_path)

    test_predictions = []
    test_y_true = []
    for images, labels in test_ds:
        preds = model.predict(images)
        test_predictions.extend(np.argmax(preds, axis=1))
        test_y_true.extend(np.argmax(labels.numpy(), axis=1))
    test_y_pred = np.array(test_predictions)
    test_y_true = np.array(test_y_true)

    if not finetune:
        return model, class_names, history, train_time_callback.times, (y_true, y_pred), (
            test_y_true, test_y_pred), summary, None, None, None, None, None

    # should_train = get_should_train_layer(model_name)
    # for layer in base_model.layers[-finetune_layers:]:
    #     if should_train(layer):
    #         layer.trainable = True
    #
    # model.compile(
    #     optimizer=finetune_optimizer,
    #     loss="categorical_crossentropy",
    #     metrics=['accuracy'] + metrics
    # )
    #
    # finetune_time_callback = TimeHistory()
    #
    # finetune_history = model.fit(train_ds, validation_data=val_ds, epochs=finetune_max_epochs,
    #                              class_weight=train_class_weights,
    #                              callbacks=[finetune_early_stopping, finetune_time_callback])
    # if model_save_path:
    #     model_file_path = os.path.join(model_save_path, "model-finetune.keras")
    #     model.save(model_file_path)
    # finetune_weights = model.get_weights()
    #
    # finetune_predictions = []
    # finetune_y_true = []
    #
    # # Iterate over the validation dataset to collect true labels and predictions
    # for images, labels in val_ds:
    #     preds = model.predict(images)
    #     finetune_predictions.extend(np.argmax(preds, axis=1))
    #     finetune_y_true.extend(np.argmax(labels.numpy(), axis=1))  # Convert one-hot to integer labels
    #
    # # Convert lists to numpy arrays
    # finetune_y_pred = np.array(finetune_predictions)
    # finetune_y_true = np.array(finetune_y_true)
    #
    # test_finetune_predictions = []
    # test_finetune_y_true = []
    # for images, labels in test_ds:
    #     preds = model.predict(images)
    #     test_finetune_predictions.extend(np.argmax(preds, axis=1))
    #     test_finetune_y_true.extend(np.argmax(labels.numpy(), axis=1))
    # test_finetune_y_pred = np.array(test_finetune_predictions)
    # test_finetune_y_true = np.array(test_finetune_y_true)
    # return model, class_names, history, train_time_callback.times, (y_true, y_pred), (
    #     test_y_true, test_y_pred), finetune_history, finetune_time_callback.times, (finetune_y_true, finetune_y_pred), (
    #     test_finetune_y_true, test_finetune_y_pred)
