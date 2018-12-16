# coding: UTF-8
# きのこデータの取得
# 必要なパッケージをインポート
import requests
import zipfile
from io import StringIO
import io
import pandas as pd
# データの前処理に必要なパッケージのインポート
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# url
mush_data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
s = requests.get(mush_data_url).content

# データの形式変換
mush_data = pd.read_csv(io.StringIO(s.decode("utf-8")), header=None)

# データに名前をつける(データを扱いやすくするため)
mush_data.columns = ["classes", "cap_shape", "cap_surface", "cap_color", "odor", "bruises",
                     "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
                     "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring",
                     "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", "veil_color",
                     "ring_number", "ring_type", "spore_print_color", "population", "habitat"]

# カテゴリー変数(色の種類など数字の大小が決められないもの)をダミー特徴量(yes or no)として変換する
mush_data_dummy = pd.get_dummies(
    mush_data[["gill_color", "gill_attachment", "odor", "cap_color"]])
# 目的変数：flg立てをする
mush_data_dummy["flg"] = mush_data["classes"].map(
    lambda x: 1 if x == "p" else 0)

# 説明変数と目的変数
X = mush_data_dummy.drop("flg", axis=1)
Y = mush_data_dummy["flg"]

# 学習データとテストデータに分ける
train_X, test_X, train_y, test_y = train_test_split(X,Y, random_state=42)

# モデルの読み込み
from sklearn.tree import DecisionTreeClassifier
# モデルの構築
model = DecisionTreeClassifier()
# モデルの学習
model.fit(train_X, train_y)

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
