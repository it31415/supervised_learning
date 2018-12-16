# coding: UTF-8
# モジュールのインポート
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_classification(
    n_samples=1000, n_features=4, n_informative=3, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# max_depthの値の範囲(1から10)
depth_list = [i for i in range(1, 11)]

# 正解率を格納するからリストを作成
accuracy = []

# 以下にコードを書いてください
# max_depthを変えながらモデルを学習
for max_depth in depth_list:
#ここに答えを書いてください
   model = DecisionTreeClassifier(max_depth=max_depth, random_state=42) 
   model.fit(train_X, train_y)
   accuracy.append(model.score(test_X, test_y))


# コードの編集はここまでです。

# グラフのプロット
plt.plot(depth_list, accuracy)
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("accuracy by changing max_depth")
plt.savefig("param_decision_tree.png")   
