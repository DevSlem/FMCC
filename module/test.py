import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from module.get_data import get_test_data
from module.write_prediction import write_predictions
from config import Config

def test(show_roc=False):
  try:
    # 만든 model로 test
    test_data_feature, test_data_label = get_test_data()
    with open('saved_model', 'rb') as f:
      model_dict = pickle.load(f)
      mod = model_dict['model']
      scaler = model_dict['scaler']
  except:
    raise('model이 없습니다. train을 먼저 해주세요')
    
  test_x_feature = scaler.transform(test_data_feature)
  test_accuracy = accuracy_score(mod.predict(test_x_feature), test_data_label)
  print(f'test_accuracy: {test_accuracy}')
      
  test_predictions = mod.predict(test_x_feature)
  write_predictions(test_predictions, f'202301ml_fmcc/{Config.team_name}_test_results.txt')

  y_scores = mod.predict_proba(test_x_feature)[:, 1] 
  fpr, tpr, _ = roc_curve(test_data_label, y_scores)
  fpr, tpr, _ = roc_curve(test_data_label, y_scores)
  
  if show_roc:
    # ROC 곡선 그리기
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()  
  else:
    return

