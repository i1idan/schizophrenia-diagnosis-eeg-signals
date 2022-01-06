import os
# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
# suppress warnings:
warnings.filterwarnings("ignore")

from datetime import datetime
import tensorflow as tf
import numpy as np
from data.load_data import load_data
from models import load_model
from utils.callbacks import get_callbacks
from sklearn.metrics import classification_report
from argparse import ArgumentParser
import random
from deep_utils import remove_create



parser = ArgumentParser()
parser.add_argument('--seed', default=1234, type=int, help="Set random seed for reproducibility")
parser.add_argument('--model-name', default='FFTCustom', help="Choose the model to train. The default is FFTCustom")
parser.add_argument('--data-path', default='./data/DATA.mat', help="Path to the Data. The default is ./data/DATA.mat")
parser.add_argument('--epochs', default=2, type=int, help="Number of training epochs, default is set to 200")
parser.add_argument('--bs', default=4, type=int, help="batch size, default is set to 4")
parser.add_argument('--checkpoints', default="./checkpoints", type=str,
                    help="Path to checkpoints, default is ./checkpoints")
parser.add_argument("--early-stopping", default=100, type=int,
                    help="early stopping patience epoch number, default is 15")
parser.add_argument("--reduce-lr", default=50, type=int, help="reduce lr patience, default is 10")
parser.add_argument("--dir-name", default='', type=str,
                    help="directory name of outputs, default is ''. If provided will overwrite existing files")

args = parser.parse_args()

# set seeds for reproducibility
np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)


def main():
    # load model
    model = load_model(model_name=args.model_name)
    print(f"[INFO] Model:{args.model_name} is loaded ...")
    model.summary()
    # load data
    (x_train, y_train), (x_test, y_test) = load_data(args.data_path, args.seed)
    # train the model
    print(f"[INFO] Started the training for model: {args.model_name} ...")
    if args.dir_name:
        dir_ = args.checkpoints + '/' + args.model_name + '/' + args.dir_name
        if os.path.exists(dir_):
            print(f"[INFO] {dir_} exists, removing it ...")
            remove_create(dir_)

    else:
        dir_ = args.checkpoints + '/' + args.model_name + "/" + '_{}'.format(
            str(datetime.now()).replace(':', '_').replace(' ', '_'))
    callbacks = get_callbacks(dir_,
                              early_stopping_p=args.early_stopping,
                              reduce_lr_patience=args.reduce_lr)
    print(f"[INFO] Training with the following arguments {args}")
    model.fit(x_train, y_train,
              epochs=args.epochs,
              batch_size=args.bs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks)
    print("[INFO] confusion matrix:!")
    print("[INFO] Loading best model:")
    model = tf.keras.models.load_model(dir_ + '/model_best')
    y_pred = np.around(model.predict(x_test))
    rep = classification_report(y_test, y_pred)
    with open(dir_ + "/conf_matrix.txt", mode='w') as f:
        f.write(rep)
    print(rep)


if __name__ == '__main__':
    main()
