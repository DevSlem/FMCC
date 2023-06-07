import os
import numpy as np
import pandas as pd
import logging

from config import Config
from module.preprocess_data import preprocess_data
from module.get_file import get_test_file, get_train_file

def get_data():
  try:
    os.mkdir(Config.folder_name)
    logging.debug(f"{Config.folder_name} 폴더를 생성했습니다.")
  except FileExistsError:
    logging.debug(f"{Config.folder_name} 폴더가 이미 존재합니다.")
  except Exception as e:
    logging.error(e)
    return
  
  try:
    logging.debug("csv 파일을 로드합니다.")
    train_data_feature = pd.read_csv(f'{Config.folder_name}/train_data_feature.csv')
    test_data_feature = pd.read_csv(f'{Config.folder_name}/test_data_feature.csv')
    
    train_label = pd.read_csv(f'{Config.folder_name}/train_data_label.csv')
    test_label = pd.read_csv(f'{Config.folder_name}/test_data_label.csv')
  except:
    logging.debug("csv 파일을 생성합니다.")
    train_data, train_label = get_train_file()
    test_data, test_label = get_test_file()

    train_data_feature = preprocess_data(train_data)
    test_data_feature = preprocess_data(test_data)
    
    train_data_feature.to_csv(f'{Config.folder_name}/train_data_feature.csv', index=False)
    test_data_feature.to_csv(f'{Config.folder_name}/test_data_feature.csv', index=False)
    
    pd.DataFrame(train_label).to_csv(f'{Config.folder_name}/train_data_label.csv', index=False)
    pd.DataFrame(test_label).to_csv(f'{Config.folder_name}/test_data_label.csv', index=False)
  else:
    train_label = train_label.values.ravel()
    test_label = test_label.values.ravel()
  finally:
    logging.debug('데이터 로드 완료')
    train_data_label = np.array(train_label)
    test_data_label = np.array(test_label)
    
  return train_data_feature, train_data_label, test_data_feature, test_data_label