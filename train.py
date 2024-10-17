import os
import random

# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

# suppress warnings:
warnings.filterwarnings("ignore")
from deep_utils import tf_set_seed
from utils.utils import save_params
from datetime import datetime
import tensorflow as tf
import numpy as np
from data.load_data import load_data
from models import load_model
from utils.callbacks import get_callbacks
from sklearn.metrics import classification_report, confusion_matrix
from argparse import ArgumentParser
from deep_utils import remove_create
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

parser = ArgumentParser()
parser.add_argument('--seed', default=1234, type=int, help="Set random seed for reproducibility")
parser.add_argument('--model-name', default='conv_lstm', help="Choose the model to train. The default is FFTCustom")
parser.add_argument('--data-path', default='./data/DATA.mat', help="Path to the Data. The default is ./data/DATA.mat")
parser.add_argument('--epochs', default=200, type=int, help="Number of training epochs, default is set to 200")
parser.add_argument('--batch-size', default=4, type=int, help="batch size, default is set to 4")
parser.add_argument('--checkpoints', default="./checkpoints", type=str,
                    help="Path to checkpoints, default is ./checkpoints")
parser.add_argument("--early-stopping", default=100, type=int,
                    help="early stopping patience epoch number, default is 15")
parser.add_argument("--reduce-lr", default=50, type=int, help="reduce lr patience, default is 10")
parser.add_argument("--dir-name", default='', type=str,
                    help="directory name of outputs, default is ''. If provided will overwrite existing files")


def main(model_name, epochs, seed, dir_name, checkpoints="./checkpoints", batch_size=4, early_stopping=100,
         reduce_lr=50, data_path="data/DATA.mat"):
    # set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    # tf_set_seed(seed)
    # load model
    model = load_model(model_name=model_name)
    print(f"[INFO] Model:{model_name} is loaded ...")
    model.summary()
    # load data
    (x_train, y_train), (x_test, y_test) = load_data(model_name=model_name,
                                                     data_path=data_path,
                                                     seed=seed)
    # train the model
    print(f"[INFO] Started the training for model: {model_name} ...")
    if dir_name:
        dir_ = checkpoints + '/' + model_name + '/' + dir_name
        if os.path.exists(dir_):
            print(f"[INFO] {dir_} exists, removing it ...")
            remove_create(dir_)
        else:
            os.makedirs(dir_, exist_ok=False)
    else:
        dir_ = checkpoints + '/' + model_name + "/" + '_{}'.format(
            str(datetime.now()).replace(':', '_').replace(' ', '_'))
        os.makedirs(dir_, exist_ok=False)
    # save params
    args = dict(model_name=model_name,
                epochs=epochs,
                seed=seed,
                dir_name=dir_name,
                checkpoints=checkpoints,
                batch_size=batch_size,
                early_stopping=early_stopping,
                reduce_lr=reduce_lr,
                data_path=data_path)
    save_params(dir_ + "/params.txt", args)
    callbacks = get_callbacks(dir_,
                              early_stopping_p=early_stopping,
                              reduce_lr_patience=reduce_lr)
    print(f"[INFO] Training with the following arguments {args}")
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks,
              shuffle=False)
    print("[INFO] confusion matrix:!")
    print("[INFO] Loading best model:")
    model = tf.keras.models.load_model(dir_ + '/model_best')
    y_pred = np.around(model.predict(x_test))

    # Save y_pred and y_true for delong_test
    pd.DataFrame(np.array([y_test.reshape(-1), y_pred.reshape(-1)]).T, columns=["y_test", "y_pred"]).to_csv(dir_ + "/y_test_pred.csv",
                                                                                          index=False)

    rep = classification_report(y_test, y_pred)
    with open(dir_ + "/classification_report.txt", mode='w') as f:
        f.write(rep)
    print(rep)

    print("[INFO] Computing Confusion matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(conf_matrix, index=["healthy", "schizophrenia"], columns=["healthy", "schizophrenia"])
    df_cm.to_csv(dir_ + '/conf_matrix.csv')
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(dir_ + '/conf_matrix.jpg')
    print("conf_matrix.jpg is successfully saved!")


if __name__ == '__main__':
    args = parser.parse_args()
    main(model_name=args.model_name, epochs=args.epochs, seed=args.seed, dir_name=args.dir_name,
         checkpoints=args.checkpoints, batch_size=args.batch_size, early_stopping=args.early_stopping,
         reduce_lr=args.reduce_lr, data_path=args.data_path)
