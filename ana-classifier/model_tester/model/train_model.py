import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from model_tester.model.get_should_train_layer import get_should_train_layer
from model_tester.metrics.f1 import f1
from model_tester.model.get_base_model import get_base_model
from model_tester.metrics.time_history import TimeHistory
import tensorflow as tf

from model_tester.model.get_input_shape import get_input_shape
from model_tester.model.get_preprocess_input import get_preprocess_input
from model_tester.model.get_top import get_top


def train_model(dst_path, model_name, data_augmentation, head="avgpool", top_dropout_rate=0.2,
                optimizer="adam", early_stopping=None, metrics=None, finetune=False, finetune_layers=20,
                finetune_optimizer="adam",
                finetune_early_stopping=None, model_save_path=None, max_epochs=20, finetune_max_epochs=10):
    if finetune_early_stopping is None:
        finetune_early_stopping = []
    if early_stopping is None:
        early_stopping = []
    shape = get_input_shape(model_name)
    img_height, img_width = shape[:2]  # Extract the first two elements
    batch_size = 32

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=dst_path,
        labels='inferred',
        subset="both",
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        validation_split=0.15,
        seed=123,
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
    # test_ds = test_ds.map(copy_red_to_green_and_blue)

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

    inputs = tf.keras.layers.Input(shape=shape)
    if data_augmentation=="":
        x=inputs
    else:
        x = data_augmentation(inputs)
    # x = inputs
    preprocess_input = tf.keras.layers.Lambda(lambda x: get_preprocess_input(model_name)(x))
    x = preprocess_input(x)

    base_model = get_base_model(model_name, shape, x)
    base_model.trainable = False
    outputs = get_top(base_model.output, num_classes, head, top_dropout_rate)
    model = tf.keras.Model(inputs, outputs, name=model_name)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"] + metrics)

    train_time_callback = TimeHistory()
    checkpoint_path = os.path.join(model_save_path, "best_weights.keras") if model_save_path else "best_weights.keras"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        verbose=1,
        monitor="val_accuracy",  # Monitor validation accuracy
        mode="max",
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, # class_weight=train_class_weights,
                        callbacks=[early_stopping, train_time_callback,model_checkpoint_callback])
    model.load_weights(checkpoint_path)
    predictions = []
    y_true = []

    for images, labels in val_ds:
        preds = model.predict(images)
        predictions.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))  # Convert one-hot to integer labels

    y_pred = np.array(predictions)
    y_true = np.array(y_true)
    if model_save_path:
        model_file_path = os.path.join(model_save_path, "model.keras")
        model.save(model_file_path)
            # Add exporting logic for inference (Stand-alone, lightweight SavedModel)
        export_path = os.path.join(model_save_path, "exported_model")
        model.export(export_path, verbose=True)  # Export lightweight TF SavedModel
        print(f"Model exported for inference to: {export_path}")

    test_predictions = []
    test_y_true = []
    # for images, labels in test_ds:
    #     preds = model.predict(images)
    #     test_predictions.extend(np.argmax(preds, axis=1))
    #     test_y_true.extend(np.argmax(labels.numpy(), axis=1))
    # test_y_pred = np.array(test_predictions)
    # test_y_true = np.array(test_y_true)

    if not finetune:
        return model, class_names, history, train_time_callback.times, (y_true, y_pred), (
        test_y_true, test_y_pred), None, None, None, None

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
        # Add ModelCheckpoint callback to save the best weights
    finetune_checkpoint_path = os.path.join(model_save_path, "best_weights_finetune.keras") if model_save_path else "best_weights_finetune.keras"
    finetune_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=finetune_checkpoint_path,
        save_best_only=True,
        verbose=1,
        monitor="val_accuracy",  # Monitor validation accuracy
        mode="max",
    )

    finetune_history = model.fit(train_ds, validation_data=val_ds, epochs=finetune_max_epochs,
                                #  class_weight=train_class_weights,
                                 callbacks=[finetune_early_stopping, finetune_time_callback, finetune_model_checkpoint_callback])
    model.load_weights(finetune_checkpoint_path)
    if model_save_path:
        model_file_path = os.path.join(model_save_path, "model-finetune.keras")
        model.save(model_file_path)
        export_path = os.path.join(model_save_path, "exported_model_finetune")
        model.export(export_path, verbose=True)  # Export lightweight TF SavedModel
        print(f"Model exported for inference to: {export_path}")

    finetune_predictions = []
    finetune_y_true = []

    # Iterate over the validation dataset to collect true labels and predictions
    for images, labels in val_ds:
        preds = model.predict(images)
        finetune_predictions.extend(np.argmax(preds, axis=1))
        finetune_y_true.extend(np.argmax(labels.numpy(), axis=1))  # Convert one-hot to integer labels

    # Convert lists to numpy arrays
    finetune_y_pred = np.array(finetune_predictions)
    finetune_y_true = np.array(finetune_y_true)

    test_finetune_predictions = []
    test_finetune_y_true = []
    # for images, labels in test_ds:
    #     preds = model.predict(images)
    #     test_finetune_predictions.extend(np.argmax(preds, axis=1))
    #     test_finetune_y_true.extend(np.argmax(labels.numpy(), axis=1))
    # test_finetune_y_pred = np.array(test_finetune_predictions)
    # test_finetune_y_true = np.array(test_finetune_y_true)
    return model, class_names, history, train_time_callback.times, (y_true, y_pred), (
    None, None), finetune_history, finetune_time_callback.times, (finetune_y_true, finetune_y_pred), (
    None, None)
