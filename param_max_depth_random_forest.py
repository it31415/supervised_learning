# coding: UTF-8
# モジュールのインポート
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_classification(
    n_samples=1000, n_features=4, n_informative=3, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# n_estimatorsの値の範囲(1から20)
n_estimators_list = [i for i in range(1, 21)]

# 正解率を格納するからリストを作成
accuracy = []

# for文内にコードを書いてください
# n_estimatorsを変えながらモデルを学習
for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(train_X, train_y)
    accuracy.append(model.score(test_X, test_y))

# グラフのプロット
plt.plot(n_estimators_list, accuracy)
plt.title("accuracy by n_estimators increasement")
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.savefig("param_max_depth_random_forest.png")
