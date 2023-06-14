import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from module.get_data import get_train_data
from module.learning_curve import plot_learning_curve

def train(do_plot=False):
  train_data_feature, train_data_label = get_train_data()
  
  scaler = StandardScaler()
  train_x_feature = scaler.fit_transform(train_data_feature)
  
  clf = SVC(kernel='rbf')
  clf.fit(train_x_feature, train_data_label)
  
  model_dict = dict(
    scaler=scaler,
    model=clf
  )
  
  # model 만들기
  print('model 생성 완료')
  with open('saved_model', 'wb') as f:
    pickle.dump(model_dict, f)
  print('model 저장 완료')
  
  if do_plot:
    plot_learning_curve(clf, train_x_feature, train_data_label)