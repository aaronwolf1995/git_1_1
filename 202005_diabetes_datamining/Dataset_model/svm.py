# Importing the libraries
import pandas

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Importing the dataset
import os
os.getcwd() # 用os包的os.getcwd函数得到当前的路径
dataset = pd.read_csv('DIA2_5.csv') # 将数据集放在当前路径夏
X = dataset.iloc[:, :-1].values # .values将data.frame转换成array
y = dataset.iloc[:, 14].values # 注意y是标签

# Splitting the dataset into the Training set and Test set
# 对train_test_split函数参数解释_我的语雀
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state = 42)# 随机种子

# Feature Scaling
# StandardScaler函数，标准化: 根据【均值】和【标准差】调整
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
数据首先fit 训练数据，然后 model 从训练数据得到必要的变换信息，
如特征方差和期望等，并保存为模型的参数，transform根据参数，
对训练数据做需要的变换。之后用在测试集上也不用在 fit 一次测试集，直接transform数据，
等于训练集和测试集所做的变换是一样的。
'''


# 设置参数字典
svm = SVC(random_state = 42)
parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 0.25, 0.5, 0.75),
'gamma': (1, 2, 3, 'auto'),'decision_function_shape': ('ovo', 'ovr'),
'shrinking': (True, False)}

# 这是一个字典

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score) # %s是占位符
    print()
# GridSearchSV函数，网格搜索找到最佳超参数组合_我的语雀
    svm = GridSearchCV(SVC(), parameters, cv=5,
    coring='%s_macro' % score)
    svm.fit(X_train, y_train)

    print("Best parameters set found on development set:") 
    print()
    print(svm.best_params_) # 挑选找到的最好参数组合
    print()
    print("Grid scores on development set:")
    print()
    # 可以通过 clf.cv_results_ 的'params'，'mean_test_score'，看一下具体的参数间不同数值的组合后得到的分数是多少
    means = svm.cv_results_['mean_test_score']
    stds = svm.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svm.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, svm.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

svm_model = SVC(kernel='rbf', C=100, gamma = 0.0001, random_state=42)
svm_model.fit(X_train, y_train)
spred = svm_model.predict(X_test)
print ('Accuracy with SVM {0}'.format(accuracy_score(spred, y_test) * 100))


# Making the Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))

round(roc_auc_score(y_test,y_pred),5)
