import numpy as np
import math

# user number
n = 943
# item number
m = 1682
r_ui = np.zeros((n, m), int)  # rating matrix
y_ui = np.zeros((n, m), int)
# observed items by user u
Ius_ = np.zeros((n, m), int)
# The tradeoff parameters
rt = 0.01
# The learning rate
lr = 0.01
# The number of latent dimensions
d = 20
# Iteration number
T = 100
# user-specific latent feature vector
Uu = np.zeros((n, d), float)
# item-specific latent feature vector
Vi = np.zeros((m, d), float)
Mi_r = np.zeros((m, 5, d), float)
# user u’s bias
bu = np.zeros(n, float)
# item i’s bias
bi = np.zeros(m, float)
u_ = None
with open("ml-100k\\u1.base", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        r_ui[int(user) - 1][int(item) - 1] = int(rating)
        if rating:
            y_ui[int(user) - 1][int(item) - 1] = 1

rm_ = np.zeros((n, m), int)
rm_p = np.zeros((n, m), float)  # 保存预测值
with open("ml-100k\\u1.test", 'r') as k:
    for line in k.readlines():
        user, item, rating, _ = line.split('	')
        rm_[int(user) - 1][int(item) - 1] = int(rating)
r = np.count_nonzero(rm_)

# number of observed ratings  p = |R|
p = np.sum(y_ui)


def initial():
    global u_
    u_ = np.sum(r_ui * y_ui) / np.sum(y_ui)

    for i in range(n):
        if np.sum(y_ui[i]) == 0:
            continue
        bu[i] = np.sum(y_ui[i] * (r_ui[i] - u_)) / np.sum(y_ui[i])
    for i in range(m):
        if np.sum(y_ui[:, i]) == 0:
            continue
        bi[i] = np.sum(y_ui[:, i] * (r_ui[:, i] - u_)) / np.sum(y_ui[:, i])

    for i in range(d):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        r3 = np.random.uniform(0, 1)
        Uu[:, i] = (r1 - 0.5) * 0.01
        Vi[:, i] = (r2 - 0.5) * 0.01
        Mi_r[:, :, i] = (r3 - 0.5) * 0.01


def ca_U_u_MPC(u, item):
    I_u_1 = I_u_2 = I_u_3 = I_u_4 = I_u_5 = 0
    M_r_1 = M_r_2 = M_r_3 = M_r_4 = M_r_5 = np.zeros(d, float)
    for i in range(m):
        if i == item:
            continue
        if r_ui[u][i] == 0:
            continue
        elif r_ui[u][i] == 1:
            M_r_1 += Mi_r[i][0]
            I_u_1 += 1
        elif r_ui[u][i] == 2:
            M_r_2 += Mi_r[i][1]
            I_u_2 += 1
        elif r_ui[u][i] == 3:
            M_r_3 += Mi_r[i][2]
            I_u_3 += 1
        elif r_ui[u][i] == 4:
            M_r_4 += Mi_r[i][3]
            I_u_4 += 1
        elif r_ui[u][i] == 5:
            M_r_5 += Mi_r[i][4]
            I_u_5 += 1
    result = np.zeros(d, float)
    if I_u_1:
        result += M_r_1 / math.sqrt(I_u_1)
    if I_u_2:
        result += M_r_2 / math.sqrt(I_u_2)
    if I_u_3:
        result += M_r_3 / math.sqrt(I_u_3)
    if I_u_4:
        result += M_r_4 / math.sqrt(I_u_4)
    if I_u_5:
        result += M_r_5 / math.sqrt(I_u_5)

    return result, I_u_1, I_u_2, I_u_3, I_u_4, I_u_5


# 计算预测后值

def predict(u, i):
    U_u_MPC = ca_U_u_MPC(u, i)[0]
    result = bu[u] + bi[i] + np.dot(Uu[u], Vi[i].T) + u_ + np.dot(U_u_MPC, Vi[i].T)
    if result > 5:
        result = 5
    elif result < 1:
        result = 1
    return result


def gradient(u, i):
    e_ui = r_ui[u][i] - predict(u, i)
    U_u_MPC, I_u_1, I_u_2, I_u_3, I_u_4, I_u_5 = ca_U_u_MPC(u, i)
    _Mi_r_1 = _Mi_r_2 = _Mi_r_3 = _Mi_r_4 = _Mi_r_5 = np.zeros(d, float)
    _u = -e_ui
    _bu = -e_ui + rt * bu[u]
    _bi = -e_ui + rt * bi[i]
    _Uu = -e_ui * Vi[i] + rt * Uu[u]
    _Vi = -e_ui * (Uu[u] + U_u_MPC) + rt * Vi[i]
    if I_u_1:
        _Mi_r_1 = -e_ui * Vi[i] / math.sqrt(I_u_1) + rt * Mi_r[i][0]
    if I_u_2:
        _Mi_r_2 = -e_ui * Vi[i] / math.sqrt(I_u_2) + rt * Mi_r[i][1]
    if I_u_3:
        _Mi_r_3 = -e_ui * Vi[i] / math.sqrt(I_u_3) + rt * Mi_r[i][2]
    if I_u_4:
        _Mi_r_4 = -e_ui * Vi[i] / math.sqrt(I_u_4) + rt * Mi_r[i][3]
    if I_u_5:
        _Mi_r_5 = -e_ui * Vi[i] / math.sqrt(I_u_5) + rt * Mi_r[i][4]
    return _u, _bu, _bi, _Uu, _Vi, _Mi_r_1, _Mi_r_2, _Mi_r_3, _Mi_r_4, _Mi_r_5


def update(u, i):
    global u_
    _u, _bu, _bi, _Uu, _Vi, _Mi_r_1, _Mi_r_2, _Mi_r_3, _Mi_r_4, _Mi_r_5 = gradient(u, i)
    # update
    u_ -= lr * _u
    bu[u] -= lr * _bu
    bi[i] -= lr * _bi
    Uu[u] -= lr * _Uu
    Vi[i] -= lr * _Vi
    Mi_r[i][0] -= lr * _Mi_r_1
    Mi_r[i][1] -= lr * _Mi_r_2
    Mi_r[i][2] -= lr * _Mi_r_3
    Mi_r[i][3] -= lr * _Mi_r_4
    Mi_r[i][4] -= lr * _Mi_r_5


def MAE(r1, r2, r_):
    return np.sum(np.abs(r1 - r2)) / r_


def RMSE(r1, r2, r_):
    return math.sqrt(np.sum((r1 - r2) * (r1 - r2)) / r_)


def model_test():
    for i in range(n):
        for j in range(m):
            if rm_[i][j] != 0:
                rm_p[i][j] = predict(i, j)
    print("MAE:%s ,RMSE:%s" % (MAE(rm_, rm_p, r), RMSE(rm_, rm_p, r)))


if __name__ == "__main__":
    initial()
    # The learning rate
    # 评价矩阵中非0索引
    non_zero_indices = np.where(r_ui != 0)
    len_index = len(non_zero_indices[0])
    for i in range(T):
        for j in range(p):
            i_k = np.random.randint(0, len_index)
            user_u = non_zero_indices[0][i_k]
            user_i = non_zero_indices[1][i_k]
            update(user_u, user_i)
            model_test()
        lr = lr * 0.9
    print(u_, bu, bi, Uu, Vi)
