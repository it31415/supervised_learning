import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
# KernelPCAをインポート
#---------------------------
from sklearn.decomposition import KernelPCA
#---------------------------

# 月形データを取得
X, y = make_moons(n_samples=100, random_state=123)

# KernelPCAクラスをインスタンス化
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
# データXをKernelPCAを用いて変換
X_kpca = kpca.fit_transform(X)

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
ax1.scatter(X[y==0, 0], X[y==0, 1], c="r")
ax1.scatter(X[y==1, 0], X[y==1, 1], c="b")
ax1.set_title("moon_data")
ax2.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], c="r")
ax2.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], c="b")
ax2.set_title("kernel_PCA")
plt.savefig('kernel_moon.png')

print(X_kpca) # 消さないでください。実行結果の確認に使います。
