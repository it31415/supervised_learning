import numpy as np

np.random.seed(39)

X = np.random.rand(8,3)

# ペアごとの平方ユークリッド距離を計算
M = np.sum((X - X[:, np.newaxis])**2, axis=2)

# カーネル行列を計算
gamma=15
K = np.exp(-gamma * M)

#---------------------------
print(K.shape)
#---------------------------

print(M) # 消さないでください。実行結果の確認に使います。
print(K) # 消さないでください。実行結果の確認に使います。
