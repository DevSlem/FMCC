import numpy as np

def get_test_file():
  file_name = open(f'202301ml_fmcc/fmcc_test_ref.txt', 'r')
  train_data_list = []
  label_list = []

  for i in file_name.readlines():
    file, label = i.strip('\n').split(' ')
    f = open(f'202301ml_fmcc/raw16k/test/' + file + '.raw', 'rb')
    file = np.fromfile(f, dtype='int16', sep="")
    
    train_data_list.append(file.astype(np.float32))
    label_list.append(0 if label[0] == 'm' else 1)
    
  return train_data_list, label_list


def get_train_file():
  file_name = open(f'202301ml_fmcc/fmcc_train.ctl', 'r')
  train_data_list = []
  label_list = []
  for i in file_name.readlines():
    i = i.strip('\n')
    f = open(f'202301ml_fmcc/raw16k/train/' + i + '.raw', 'rb')
    file = np.fromfile(f, dtype='int16', sep="")
    
    train_data_list.append(file.astype(np.float32))
    label_list.append(0 if i[0] == 'M' else 1)
    
  return train_data_list, label_list