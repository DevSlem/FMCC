import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from module.get_data import get_train_data

def train():
  train_data_feature, train_data_label = get_train_data()
  
  scaler = StandardScaler()
  train_x_pca = scaler.fit_transform(train_data_feature)
  
  clf = SVC(kernel='rbf', probability=True)
  clf.fit(train_x_pca, train_data_label)
  
  model_dict = dict(
    # scaler=scaler,
    model=clf
  )
  
  # model 만들기
  print('model 생성 완료')
  with open('saved_model', 'wb') as f:
    pickle.dump(model_dict, f)
  print('model 저장 완료')