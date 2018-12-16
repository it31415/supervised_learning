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

# 以下にコードを記述してください。
# モデルの読み込み
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# モデルの構築
model1 = RandomForestClassifier()
model2 = DecisionTreeClassifier()
# モデルの学習
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
