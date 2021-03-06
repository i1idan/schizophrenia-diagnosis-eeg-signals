"""
This module contains a function that provides callbacks.
"""
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from datetime import datetime


def get_callbacks(checkpoint_dir,
                  early_stopping_p,
                  reduce_lr_patience,
                  **kwargs):
    """
    This function use some callbacks from tensorflow.python.keras.callbacks

    Parameters
    ----------
    checkpoint_dir: str ; path to save the model file.
    early_stopping_p: int ; number of epochs with no improvement after which training will be stopped.
    **kwargs

    Returns
    -------
    checkpoint: a tensorflow.python.keras.callbacks.ModelCheckpoint instance
    reduce_lr: a tensorflow.python.keras.callbacks.ReduceLROnPlateau instance
    early_stopping: a tensorflow.python.keras.callbacks.EarlyStopping instance
    """
    checkpoint = ModelCheckpoint(filepath=checkpoint_dir + '/model_best',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=False,
                                 )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,  # new_lr = lr * factor
                                  patience=reduce_lr_patience,  # number of epochs with no improvment
                                  min_lr=1e-5,  # lower bound on the learning rate
                                  mode='min',
                                  verbose=1
                                  )

    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_p, verbose=1)

    tensorboard = TensorBoard(
        log_dir=checkpoint_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=True)
    csv_logger = CSVLogger(
        checkpoint_dir + "/log.csv",
        append=True)
    return checkpoint, early_stopping, tensorboard, reduce_lr, csv_logger
