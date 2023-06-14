import os
import numpy as np
import pandas as pd
import logging

from config import Config
from module.preprocess_data import preprocess_data
from module.get_file import get_train_file, get_test_file, get_eval_file
from module.util import try_make_dir

def get_train_data():
  try_make_dir(Config.folder_name)
  
  try:
    print("csv 파일을 로드합니다.")
    train_data_feature = pd.read_csv(f'{Config.folder_name}/train_data_feature.csv')
    train_label = pd.read_csv(f'{Config.folder_name}/train_data_label.csv')
  except:
    print("csv 파일 생성중 ....")
    train_data, train_label = get_train_file()

    print("데이터 전처리를 시작합니다.")
    train_data_feature = preprocess_data(train_data)
    print("데이터 전처리를 완료하였습니다.")
    
    train_data_feature.to_csv(f'{Config.folder_name}/train_data_feature.csv', index=False)
    
    pd.DataFrame(train_label).to_csv(f'{Config.folder_name}/train_data_label.csv', index=False)
    print("csv 파일을 생성하였습니다.")
  else:
    train_label = train_label.values.ravel()
  finally:
    print('데이터 로드 완료')
    train_data_label = np.array(train_label)
  
  return train_data_feature, train_data_label

def get_eval_data():
  try_make_dir(Config.folder_name)
  
  try:
    print("csv 파일을 로드합니다.")
    eval_data_feature = pd.read_csv(f'{Config.folder_name}/eval_data_feature.csv')
  except:
    print("csv 파일 생성중 ....")
    eval_data = get_eval_file()

    print("데이터 전처리를 시작합니다.")
    eval_data_feature = preprocess_data(eval_data)
    print("데이터 전처리를 완료하였습니다.")
    
    eval_data_feature.to_csv(f'{Config.folder_name}/eval_data_feature.csv', index=False)
    
    print("csv 파일을 생성하였습니다.")
  finally:
    print('데이터 로드 완료')
    
  return eval_data_feature

def get_test_data():
  try_make_dir(Config.folder_name)
  
  try:
    print("csv 파일을 로드합니다.")
    test_data_feature = pd.read_csv(f'{Config.folder_name}/test_data_feature.csv')
    test_label = pd.read_csv(f'{Config.folder_name}/test_data_label.csv')
  except:
    print("csv 파일 생성중 ....")
    test_data, test_label = get_test_file()

    print("데이터 전처리를 시작합니다.")
    test_data_feature = preprocess_data(test_data)
    print("데이터 전처리를 완료하였습니다.")
    
    test_data_feature.to_csv(f'{Config.folder_name}/test_data_feature.csv', index=False)
    
    pd.DataFrame(test_label).to_csv(f'{Config.folder_name}/test_data_label.csv', index=False)
    print("csv 파일을 생성하였습니다.")
  else:
    test_label = test_label.values.ravel()
  finally:
    print('데이터 로드 완료')
    test_data_label = np.array(test_label)
    
  return test_data_feature, test_data_label