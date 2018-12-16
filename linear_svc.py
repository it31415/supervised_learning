# coding: UTF-8
# パッケージをインポート
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# データの生成
X, y = make_classification(n_samples=100, n_features=2,
                           n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# 以下にコードを記述してください
# モデルの構築
model = LinearSVC()

# train_Xとtrain_yを使ってモデルに学習させる
#ここに答えをかいてください
model.fit(train_X, train_y)
# test_Xとtest_yを用いたモデルの正解率を出力
#ここに答えを書いてください
print(model.score(test_X, test_y))

pred_y = model.predict(test_X)
accuracy = model.score(test_X, test_y)

# 適合率、再現率、F1
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

precision = precision_score(test_y, pred_y, average = None)
recall = recall_score(test_y, pred_y, average = None)
f1 = f1_score(test_y, pred_y, average = None)

print("test_y: {}".format(test_y))
print("pred_y: {}".format(pred_y))
print(test_y == pred_y)
print("accuracy: {}".format(accuracy))
print("precision: {}".format(precision))
print("recall: {}".format(recall))
print("f1: {}".format(f1))

# コードの編集はここまでです。
# 生成したデータをプロット
plt.scatter(X[:, 0], X[:, 1], c=y, marker=".",
            cmap=matplotlib.cm.get_cmap(name="bwr"), alpha=0.7)

# 学習して導出した識別境界線をプロット
Xi = np.linspace(-10, 10)
Y = -model.coef_[0][0] / model.coef_[0][1] * Xi - model.intercept_ / model.coef_[0][1]
plt.plot(Xi, Y)

# グラフのスケールを調整
plt.xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
plt.ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
plt.axes().set_aspect("equal", "datalim")
# グラフにタイトルを設定する
plt.title("classification data using LinearSVC")
# x軸、y軸それぞれに名前を設定する
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.savefig("linear_svc.png")
