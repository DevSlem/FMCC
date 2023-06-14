import numpy as np

from config import Config

def get_test_file():
  with open(Config.test_file_list, 'r') as f:
    
    test_data_list = []
    label_list = []

    for i in f.readlines():
      file, label = i.strip('\n').split(' ')
      
      with open(f'raw16k/test/' + file + '.raw', 'rb') as f:
        file = np.fromfile(f, dtype='int16', sep="")
        test_data_list.append(file.astype(np.float32))
        label_list.append(0 if label[0] == 'm' else 1)
          
  return test_data_list, label_list

def get_eval_file():
  file_name = open(Config.eval_file_list, 'r')
  eval_data_list = []
  
  for i in file_name.readlines():
    file = i.strip('\n')
    
    with open(Config.eval_file_dir + file + '.raw', 'rb') as f:
      file = np.fromfile(f, dtype='int16', sep="")
      eval_data_list.append(file.astype(np.float32))
  file_name.close()
  return eval_data_list


def get_train_file():
  file_name = open(Config.train_file_list, 'r')
  train_data_list = []
  label_list = []
  
  for i in file_name.readlines():
    i = i.strip('\n')
    with open(f'raw16k/train/' + i + '.raw', 'rb') as f:
      file = np.fromfile(f, dtype='int16', sep="")
      train_data_list.append(file.astype(np.float32))
      label_list.append(0 if i[0] == 'M' else 1)
  
  file_name.close()
  return train_data_list, label_list