# coding: UTF-8
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
train_X, test_X, train_y, test_y = train_test_split(
    data.data, data.target, random_state=42)
print(data)

# パラメーターの値の候補を設定
#param = {
#    "C": [10 ** -i for i in range(5)],
#    "kernel": ["linear", "poly", "rbf", "sigmoid"],
#    "random_state": [42]
#}

# 学習器を構築(ここではパラメーターを調整しない)
#svm = SVC()
#print("before search")

# グリッドサーチ実行
#clf = GridSearchCV(svm, param)
#clf.fit(train_X, train_y)

# パラメーターサーチ結果の取得
#best_param = clf.best_params_

# 比較のため調整なしのsvmに対しても学習させ正解率を比較
#svm.fit(train_X, train_y)
#print("調整なしsvm:{}\n調整ありsvm:{}\n最適パラメーター:{}".format(
#    svm.score(test_X, test_y), clf.score(test_X, test_y), best_param))
