import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import random

from module.get_data import get_data


if __name__ == "__main__":
    # seed 고정
    random.seed(0)
    np.random.seed(0)
    
    train_data_feature, train_data_label, test_data_feature, test_data_label = get_data()
    
    # Apply scaling for PCA
    scaler = StandardScaler()
    train_x_pca = scaler.fit_transform(train_data_feature)
    test_x_pca = scaler.fit_transform(test_data_feature)
    
    # Fit an SVM model
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(train_x_pca, train_data_label)

    train_accuracy = accuracy_score(clf.predict(train_x_pca), train_data_label)
    print(train_accuracy)
    
    test_accuracy = accuracy_score(clf.predict(test_x_pca), test_data_label)
    print(test_accuracy)

    y_scores = clf.predict_proba(test_x_pca)[:, 1] 
    fpr, tpr, thresholds = roc_curve(test_data_label, y_scores)

    fpr, tpr, thresholds = roc_curve(test_data_label, y_scores)

    # ROC 곡선 그리기
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()