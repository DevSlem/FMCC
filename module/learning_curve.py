import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(clf, train_x_feature, train_data_label):
  train_sizes, _, test_scores = learning_curve(clf, train_x_feature, train_data_label, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)

  # Plot the learning curve
  plt.figure()
  plt.title("SVM Learning Curve")
  plt.xlabel("Training Examples")
  plt.ylabel("Score")
  plt.grid()
  
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")

  plt.legend(loc="best")
  plt.show()