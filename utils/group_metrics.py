from collections import defaultdict
import pandas as pd
import numpy as np


def get_mean_std(csv_lists,
                 arguments=("accuracy", "loss", "val_accuracy", "val_loss"),
                 operators=(max, min, max, min)
                 ):
    metrics = defaultdict(list)
    print(f"[INFO] Extracting values from csv files...")
    for csv in csv_lists:
        print(f"[INFO] Getting the values of file {csv}")
        csv_file = pd.read_csv(csv)
        for metric, op in zip(arguments, operators):
            val = op(csv_file[metric])
            metrics[metric].append(val)

    metrics = {metric: {"std": round(np.std(val_list), 4), "mean": round(np.mean(val_list), 4)} for metric, val_list in
               metrics.items()}
    return metrics


if __name__ == '__main__':
    metrics = get_mean_std(
        csv_lists=["/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_37_26.379079/log.csv",
                   "/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_09_40.034447/log.csv",
                   "/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_07_52.880768/log.csv",
                   "/home/ai/projects/schizo/checkpoints/FFTCustom/_2022-01-06_13_06_43.842120/log.csv",
                   ])
    print(metrics)
