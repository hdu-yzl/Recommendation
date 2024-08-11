import numpy as np
import math
from Evaluation import ca_test

# learning rate
lr = 0.01
# dimension of latent vector
d = 20
# sampling size
p = 3
# regularization term
rt = 0.01
# number of iteration
T = 100
# number of users
n = 943
# item number
m = 1682
r_ui = np.zeros((n, m), int)  # rating matrix
y_ui = np.zeros((n, m), int)  # {-1,1}
y_ui_ = np.zeros((n, m), int)  # {0,1}
# the set of positive samples
S_P = None
# the set of negative samples
S_N = None
# predicted preference of user u to item i
r_ui_ = np.zeros((n, m), float)
# user-specific latent feature vector
Uu = np.zeros((n, d), float)
# item-specific latent feature vector
Vi = np.zeros((m, d), float)
Mi_r = np.zeros((m, 5, d), float)
# user u’s bias
bu = np.zeros(n, float)
# item i’s bias
bi = np.zeros(m, float)
# average rating value
ru_ = None
# global average rating value
u_ = None
with open("ml-100k/u1.base.explicit", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        r_ui[int(user) - 1][int(item) - 1] = int(rating)
        if rating:
            y_ui_[int(user) - 1][int(item) - 1] = 1

r_ui_t_yui = np.zeros((n, m))  # 测试数据的yui矩阵
r_ui_p_ui = np.zeros((n, m))  # 预测矩阵
U_te = np.zeros(n)  # 测试用户
U_te_num = 0  # 测试用户数
K = 5

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


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


def initial():
    global S_P, S_N
    global u_
    u_ = np.sum(r_ui) / len(np.where(r_ui != 0)[0])
    for i in range(n):
        if np.sum(y_ui_[i]) == 0:
            continue
        bu[i] = np.sum(y_ui_[i] * (r_ui[i] - u_)) / np.sum(y_ui_[i])
    for i in range(m):
        if np.sum(y_ui_[:, i]) == 0:
            continue
        bi[i] = np.sum(y_ui_[:, i] * (r_ui[:, i] - u_)) / np.sum(y_ui_[:, i])

    for i in range(d):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        r3 = np.random.uniform(0, 1)
        Uu[:, i] = (r1 - 0.5) * 0.01
        Vi[:, i] = (r2 - 0.5) * 0.01
        Mi_r[:, :, i] = (r3 - 0.5) * 0.01
    global ru_
    ru_ = np.zeros(n, float)

    for i in range(n):
        if np.sum(y_ui_[i]):
            ru_[i] = np.sum(r_ui[i]) / np.sum(y_ui_[i])
    for i in range(n):
        for j in range(m):
            if r_ui[i][j] == 0:
                continue
            if r_ui[i][j] >= ru_[i]:
                y_ui[i][j] = 1
                if S_P is None:
                    S_P = np.array((i, j, 1))
                else:
                    S_P = np.vstack([S_P, [i, j, 1]])

            else:
                y_ui[i][j] = -1
                if S_N is None:
                    S_N = np.array((i, j, -1))
                else:
                    S_N = np.vstack([S_N, [i, j, -1]])


def predict(u, i):
    U_u_MPC = ca_U_u_MPC(u, i)[0]
    return bu[u] + bi[i] + np.dot(Uu[u], Vi[i].T) + u_ + np.dot(U_u_MPC, Vi[i].T)


def gradient(u, i, yui):
    rui_ = predict(u, i)
    #print(rui_)
    U_u_MPC, I_u_1, I_u_2, I_u_3, I_u_4, I_u_5 = ca_U_u_MPC(u, i)
    _Mi_r_1 = _Mi_r_2 = _Mi_r_3 = _Mi_r_4 = _Mi_r_5 = np.zeros(d, float)
    e_ui = (1 - sigmoid(rui_ * yui)) * yui
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


def update(u, i, yui):
    global u_
    _u, _bu, _bi, _Uu, _Vi, _Mi_r_1, _Mi_r_2, _Mi_r_3, _Mi_r_4, _Mi_r_5 = gradient(u, i, yui)
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


r_ui_implicit = []
with open("ml-100k/u1.base.implicit", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        r_ui_implicit.append([int(user) - 1, int(item) - 1, -1])

def train():
    global lr, S_N, S_P
    for t in range(T):
        S_N_new = S_N
        for i in range(len(r_ui_implicit)):
            rd = np.random.randint(0, 5)
            if rd < 4:
                S_N_new = np.vstack([S_N_new, r_ui_implicit[i]])
        S = np.vstack([S_P, S_N_new])
        len_S = len(S)
        for t2 in range(len_S):
            k1 = np.random.randint(0, len_S)
            u, i, yui = S[k1]
            gradient(u, i, yui)
            update(u, i, yui)
        model_test()
        lr *= 0.9



def load_test_data():
    global ru_
    global U_te_num
    with open("ml-100k\\u1.test", 'r') as k:
        for line in k.readlines():
            user, item, rating, z = line.split('	')
            if int(rating) >= ru_[int(user) - 1]:
                r_ui_t_yui[int(user) - 1][int(item) - 1] = 1
            else:
                r_ui_t_yui[int(user) - 1][int(item) - 1] = -1

    for i in range(n):
        if np.count_nonzero(r_ui_t_yui[i]):
            U_te[i] = 1
            U_te_num += 1

y_ui_train = np.zeros((n,m))
with open("ml-100k/u1.base.explicit", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        y_ui_train[int(user) - 1][int(item) - 1] = 1

def model_test():
    for user in range(n):
        for item in range(m):
            if y_ui_train[user][item] == 0:
                r_ui_p_ui[user][item] = predict(user, item)
            else:
                r_ui_p_ui[user][item] = np.NINF
    np.save('r_ui_p_ui', r_ui_p_ui)
    ca_test(r_ui_p_ui, r_ui_t_yui, K, U_te, U_te_num)


if __name__ == "__main__":
    initial()
    load_test_data()
    train()
