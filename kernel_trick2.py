import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# データが円状に分布するデータを取得
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

# ペアごとの平方ユークリッド距離を計算
M = np.sum((X - X[:, np.newaxis])**2, axis=2)

# 対称カーネル行列を計算。γの値は15にしてください。
gamma=15
K = np.exp(-gamma * M)
    
# カーネル行列から固有対を取得。 numpy.linalg.eighはそれらを固有値の昇順で返す
eigvals, eigvecs = np.linalg.eigh(K)
# 上位k個の固有ベクトル(射影されたサンプル)を収集
W = np.c_[eigvecs[:,-1], eigvecs[:,-2]]

# 線形分離可能なデータが得られます。
X_kpca = K.dot(W)

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
ax1.scatter(X[y==0, 0], X[y==0, 1], color="r", marker="^")
ax1.scatter(X[y==1, 0], X[y==1, 1], color="b", marker="o")
ax2.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color="r", marker="^")
ax2.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color="b", marker="o")
ax1.set_title("circle_data")
ax2.set_title("kernel_pca")
plt.savefig('kernel_trick2.png')

print(M) # 消さないでください。実行結果の確認に使います。
print(K) # 消さないでください。実行結果の確認に使います。
print(W) # 消さないでください。実行結果の確認に使います。
print(X_kpca) # 消さないでください。実行結果の確認に使います。
