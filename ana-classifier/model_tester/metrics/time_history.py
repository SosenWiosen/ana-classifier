import time

import tensorflow as tf


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = None
        self.epoch_start_time = None

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)
