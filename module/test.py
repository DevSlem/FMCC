import pickle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from module.get_data import get_test_data, get_eval_data
from module.write_prediction import write_predictions
from config import Config

def eval():
  try:
    eval_data_feature = get_eval_data()
    with open(f'{Config.results_dir}/saved_model', 'rb') as f:
      model_dict = pickle.load(f)
      mod = model_dict['model']
      scaler = model_dict['scaler']
  except:
    raise Exception('model이 없습니다. train을 먼저 해주세요')
    
  eval_x_feature = scaler.transform(eval_data_feature)
      
  eval_predictions = mod.predict(eval_x_feature)
  write_predictions(eval_predictions, f'{Config.results_dir}/{Config.team_name}_test_results.txt')

def test():
  try:
    test_data_feature, test_data_label = get_test_data()
    with open(f'{Config.results_dir}/saved_model', 'rb') as f:
      model_dict = pickle.load(f)
      mod = model_dict['model']
      scaler = model_dict['scaler']
  except:
    raise Exception('model이 없습니다. train을 먼저 해주세요')
    
  test_x_feature = scaler.transform(test_data_feature)
  test_accuracy = accuracy_score(mod.predict(test_x_feature), test_data_label)
  print(f"test accuracy: {test_accuracy}")

