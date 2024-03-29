from collections import defaultdict
import pandas as pd
import numpy as np


def get_mean_std(csv_lists,
                 arguments=("accuracy", "loss", "val_accuracy", "val_loss"),
                 # operator=(max, min, max, min),
                 get_min_index_op=np.argmin,
                 get_min_index_metric="val_loss",
                 ):
    metrics = defaultdict(list)
    print(f"[INFO] Extracting values from csv files...")
    for csv in csv_lists:
        print(f"[INFO] Getting the values of file {csv}")
        df = pd.read_csv(csv)
        index = get_min_index_op(df[get_min_index_metric])
        for metric in arguments:
            val = df[metric][index]
            metrics[metric].append(val)

    metrics = {metric: {"std": round(np.std(val_list), 4), "mean": round(np.mean(val_list), 4)} for metric, val_list in
               metrics.items()}
    return metrics


def get_conf_mean_std(conf_matrices):
    metrics = defaultdict(list)
    print(f"[INFO] Extracting values from confusion matrices")
    for conf in conf_matrices:
        df = pd.read_csv(conf, index_col=0).values
        print(f"[INFO] Getting the values of df: {conf}")
        tp = df[1, 1]
        tn = df[0, 0]
        fn = df[0, 1]
        fp = df[1, 0]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        if str(specificity) == 'nan':
            specificity = 0
        if str(sensitivity) == "nan":
            sensitivity = 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = (2 * tp) / (2 * tp + fp + fn)
        metrics["sensitivity"].append(sensitivity)
        metrics["specificity"].append(specificity)
        metrics["accuracy"].append(accuracy)
        metrics["f1_score"].append(f1_score)

    metrics = {metric: {"std": round(np.std(val_list), 4), "mean": round(np.mean(val_list), 4)} for metric, val_list in
               metrics.items()}
    print(f'[INFO] Successfully Extracted {metrics.keys()}')
    return metrics


if __name__ == '__main__':
    # metrics = get_mean_std(
    #     csv_lists=["/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_37_26.379079/log.csv",
    #                "/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_09_40.034447/log.csv",
    #                "/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_07_52.880768/log.csv",
    #                "/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_06_43.842120/log.csv",
    #                ])
    # print(metrics)
    import os

    multi_train = 10
    model_names = ["WaveletCustom", "FFTCustom", "Transformer", "conv_lstm"]
    checkpoints = "../checkpoints"
    # for model_name in model_names:
    #     print(f"[INFO] Processing Model: {model_name}")
    #     csv_files = [os.path.join(checkpoints, model_name, f"{n}", "log.csv") for n in range(multi_train)]
    #     metrics = get_mean_std(csv_files,
    #                            arguments=("accuracy", "loss", "val_accuracy", "val_loss"))
    #     print(model_name, "\n", metrics)
    for model_name in model_names:
        print(f"[INFO] Processing {model_name}")
        conf_matrixes = [os.path.join(checkpoints, model_name, f"{n}", "conf_matrix.csv") for n in range(multi_train)]

        metrics = get_conf_mean_std(conf_matrixes)
        print(model_name, "\n", metrics)