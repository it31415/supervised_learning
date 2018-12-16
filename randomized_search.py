#coding:UTF-8
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
train_X, test_X, train_y, test_y = train_test_split(
    data.data, data.target, random_state=42)

# パラメーターの値の候補を設定
param = {
    # 0から100までの一様確率変数(どの数も全て同じ確率で現れる)を定義
    "C": stats.uniform(loc=0.0, scale=100.0),
    # 乱数で選ぶ必要がないものはリストで指定
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "random_state": [42]
}

# 学習器を構築(ここではパラメーターを調整しない)
svm = SVC()

# ランダムサーチ実行
clf = RandomizedSearchCV(svm, param)
clf.fit(train_X, train_y)

# パラメーターサーチ結果の取得
best_param = clf.best_params_

# 比較のため調整なしのsvmに対しても学習させ正解率を比較
svm.fit(train_X, train_y)
print("調整なしsvm:{}\n調整ありsvm:{}\n最適パラメーター:{}".format(
    svm.score(test_X, test_y), clf.score(test_X, test_y), best_param))
