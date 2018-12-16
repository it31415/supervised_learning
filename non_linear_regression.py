# coding: UTF-8
# パッケージをインポート
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles

# データの生成
X, y = make_gaussian_quantiles(
    n_samples=1000, n_classes=2, n_features=2, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# 以下にコードを記述してください
# モデルの構築
#ここに答えを書いてください
model1 = SVC()
model2 = LinearSVC()
# train_Xとtrain_yを使ってモデルに学習させる
#ここに答えを書いてください
model1.fit(train_X, train_y)
model2.fit(train_X, train_y)

pred_y1 = model1.predict(test_X)
pred_y2 = model2.predict(test_X)

# 正解率

accuracy1 = model1.score(test_X, test_y)
accuracy2 = model2.score(test_X, test_y)

# 適合率、再現率、F1
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

precision1 = precision_score(test_y, pred_y1, average = None)
precision2 = precision_score(test_y, pred_y2, average = None)

recall1 = recall_score(test_y, pred_y1, average = None)
recall2 = recall_score(test_y, pred_y2, average = None)

f1_1 = f1_score(test_y, pred_y1, average = None)
f1_2 = f1_score(test_y, pred_y2, average = None)

print("test_y: {}".format(test_y))
print("pred_y1: {}".format(pred_y1))
print(test_y == pred_y1)
print("pred_y2: {}".format(pred_y2))
print(test_y == pred_y2)
print("accuracy1: {}".format(accuracy1))
print("accuracy2: {}".format(accuracy2))
print("precision1: {}".format(precision1))
print("precision2: {}".format(precision2))
print("recall1: {}".format(recall1))
print("recall2: {}".format(recall2))
print("f1_1: {}".format(f1_1))
print("f1_2: {}".format(f1_2))

# 生成したデータをプロット
plt.scatter(X[:, 0], X[:, 1], c=y, marker=".", cmap=matplotlib.cm.get_cmap(name="bwr"), alpha=0.7)
# グラフにタイトルを設定する
plt.title("classification data using non Logistic Regression")
# x軸、y軸それぞれに名前を設定する
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.savefig("non_linear_regression.png")
