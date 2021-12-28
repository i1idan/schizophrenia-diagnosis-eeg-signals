from datetime import datetime
import tensorflow as tf
import numpy as np
from data.load_data import load_data
from models import load_model
from utils.callbacks import get_callbacks
from sklearn.metrics import classification_report
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--seed', default=1234, help="Set random seed for reproducibility")
parser.add_argument('--model-name', default='FFTCustom', help="Choose the model to train. The default is FFTCustom")
parser.add_argument('--data-path', default='./data/DATA.mat', help="Path to the Data. The default is ./data/DATA.mat")
parser.add_argument('--epochs', default=200, type=int, help="Number of training epochs, default is set to 200")
parser.add_argument('--bs', default=4, type=int, help="batch size, default is set to 4")
parser.add_argument('--checkpoints', default="./checkpoints", type=str,
                    help="Path to checkpoints, default is ./checkpoints")
parser.add_argument("--early-stopping", default=50, type=int,
                    help="early stopping patience epoch number, default is 15")
parser.add_argument("--reduce-lr", default=25, type=int, help="reduce lr patience, default is 10")
args = parser.parse_args()

# set seeds for reproducibility
tf.random.set_seed(args.seed)
np.random.seed(args.seed)

# load data
(x_train, y_train), (x_test, y_test) = load_data(args.data_path, args.seed)

# load model
model = load_model(model_name=args.model_name)
print(f"[INFO] Model:{args.model_name} is loaded ...")
model.summary()

# train the model
print(f"[INFO] Started the training for model: {args.model_name} ...")
dir_ = args.checkpoints + '/' + args.model_name + "/" + '_{}'.format(
    str(datetime.now()).replace(':', '_').replace(' ', '_'))
callbacks = get_callbacks(dir_,
                          early_stopping_p=args.early_stopping,
                          reduce_lr_patience=args.reduce_lr)
history = model.fit(x_train, y_train,
                    epochs=args.epochs,
                    batch_size=args.bs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)

print("[INFO] confusion matrix:!")
model = tf.keras.models.load_model(dir_ + '/model_best')
y_pred = np.around(model.predict(x_test))
rep = classification_report(y_test, y_pred)
print(rep)
