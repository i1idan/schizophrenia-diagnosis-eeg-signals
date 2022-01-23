import matplotlib.pyplot as plt
import pandas as pd

def plot_history(train_acc, val_acc, train_loss, val_loss, figsize=(10, 5), 
                 dpi=300, acc_path='accuracy.png', loss_path='loss.png'):
  plt.figure(figsize=figsize)
  plt.plot(train_acc)
  plt.plot(val_acc)
  plt.title("Model Accuracy")
  plt.ylabel("Accuracy")
  plt.xlabel("Epoch")
  plt.savefig(acc_path, dpi=dpi)
  print(f"[INFO] Accuracy Figure saved to: {acc_path}")
  plt.figure(figsize=figsize)
  plt.plot(train_loss)
  plt.plot(val_loss)
  plt.title("Model Loss")       
  plt.ylabel("Loss") 
  plt.xlabel("Epoch")
  plt.savefig(loss_path, dpi=dpi)
  print(f"[INFO] Loss Figure saved to: {loss_path}")

def plot_loss_from_logs(csv_lists):
  print(f"[INFO] Extracting values from csv files...")
  for i, csv in enumerate(csv_lists):
      print(f"[INFO] Getting the values of file {csv}")
      csv_file = pd.read_csv(csv)
      train_acc = csv_file['accuracy']
      train_loss = csv_file['loss']
      val_acc = csv_file['val_accuracy']
      val_loss = csv_file['val_loss']
      # for now save each run independently
      plot_history(train_acc, val_acc, train_loss, val_loss, figsize=(10, 5), 
                   dpi=300, acc_path=f'accuracy_{i}.png',
                   loss_path=f'loss_{i}.png')
      
      
if __name__ == '__main__':
  
  import os

  base_path = "/content/drive/MyDrive/FFTCustom"
  csv_lists = []
  for run_dir in os.listdir(base_path):
    run_dir_fullpath = os.path.join(base_path, run_dir)
    log_files = os.listdir(run_dir_fullpath)
    log_filename = "log.csv"
    csv_lists.append(os.path.join(run_dir_fullpath, log_filename))
    plot_loss_from_logs(csv_lists=csv_lists)
