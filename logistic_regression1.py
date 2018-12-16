# coding: UTF-8
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# データの生成
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)
# データを学習に使う分と評価の分に分ける
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# モデルの構築
model = LogisticRegression(random_state=42)

# train_Xとtrain_yを使ってモデルに学習させる
#ここにコードを書いてください
model.fit(train_X, train_y)
# test_Xに対するモデルの分類予測結果
from sklearn.metrics import accuracy_score
pred_y = model.predict(test_X)
score = model.score(test_X, test_y)
accuracy_score(test_y, pred_y)

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
