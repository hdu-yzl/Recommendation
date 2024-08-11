import numpy as np

n = 943
m = 1682
y_ui_train = np.zeros((n, m), int)
y_ui_test = np.zeros((n, m), int)
r_ui_predict = np.zeros((n, m), float)
with open("ml-100k/u1.base.OCCF", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        y_ui_train[int(user) - 1][int(item) - 1] = 1

with open("ml-100k/u1.test.OCCF", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        y_ui_test[int(user) - 1][int(item) - 1] = 1


# 计算预测矩阵
def ca_predict():
    u = np.sum(y_ui_train) / n / m
    for i in range(n):
        r_ui_predict[i, :] = np.sum(y_ui_train, axis=0) / n - u
    for i in range(n):  # 要把训练集中用户评价过的物品修改为无限小
        for j in range(m):
            if y_ui_train[i][j]:
                r_ui_predict[i][j] = np.NINF


def Pre_k(I_u_re, I_u_te, k, U_te, U_te_num):
    sorted_indices = np.argsort(-I_u_re, axis=1)  # 从大到小排序
    top_k_indices = sorted_indices[:, :k]
    pre_k = 0
    Pre_k_u = []
    for user in range(n):
        sum_x = 0
        if U_te[user] == 0:
            continue
        for item in range(k):
            if I_u_te[user][top_k_indices[user][item]] == 1:
                sum_x += 1
        pre_k += sum_x
        Pre_k_u.append(sum_x / k)

    return pre_k / k / U_te_num, Pre_k_u


def Rec_k(I_u_re, I_u_te, k, U_te, U_te_num):
    sorted_indices = np.argsort(-I_u_re, axis=1)
    top_k_indices = sorted_indices[:, :k]
    rec_k = 0
    Rec_k_u = []
    for user in range(n):
        if U_te[user] == 0:
            continue
        sum_x = 0
        for item in range(k):
            if I_u_te[user][top_k_indices[user][item]] == 1:
                sum_x += 1
        if np.count_nonzero(I_u_te[user]):
            rec_k += sum_x / np.count_nonzero(I_u_te[user])
            Rec_k_u.append(sum_x / np.count_nonzero(I_u_te[user]))
        else:
            Rec_k_u.append(0)
    return rec_k / U_te_num, Rec_k_u


def F1_k(I_u_re, I_u_te, k, U_te, U_te_num):
    Pre_ks = Pre_k(I_u_re, I_u_te, k, U_te, U_te_num)[1]
    Rec_ks = Rec_k(I_u_re, I_u_te, k, U_te, U_te_num)[1]
    f1_k = 0
    for i in range(len(Pre_ks)):
        if Pre_ks[i] + Rec_ks[i]:
            f1_k += (Pre_ks[i] * Rec_ks[i]) / (Pre_ks[i] + Rec_ks[i]) * 2

    return f1_k / U_te_num


def NDCG_k(I_u_re, I_u_te, k, U_te, U_te_num):
    sorted_indices = np.argsort(-I_u_re, axis=1)  # 从小到大排序
    top_k_indices = sorted_indices[:, :k]
    dcg_k_u = 0
    for user in range(n):
        if U_te[user] == 0:
            continue
        zu = 0  # 计算每个用户的zu
        for i in range(min(k, np.count_nonzero(I_u_te[user]))):
            zu += 1. / np.log(2 + i)
        dcg_k = 0
        for i in range(k):
            if I_u_te[user][top_k_indices[user][i]] == 1:
                dcg_k += 1.0 / np.log(2 + i)
        dcg_k_u += dcg_k / zu

    return dcg_k_u / U_te_num


def call_k(I_u_re, I_u_te, k, U_te, U_te_num):
    sorted_indices = np.argsort(-I_u_re, axis=1)
    top_k_indices = sorted_indices[:, :k]
    call_k = 0
    for user in range(n):
        if U_te[user] == 0:
            continue
        num = 0
        for i in range(k):
            if I_u_te[user][top_k_indices[user][i]] == 1:
                num += 1
        if num >= 1:
            call_k += 1
    return call_k / U_te_num


def MRR(I_u_re, I_u_te, k, U_te, U_te_num):
    sorted_indices = np.argsort(-I_u_re, axis=1)  # 从大到小排序
    top_k_indices = sorted_indices[:, :k]
    RR_u = 0
    for user in range(n):
        if U_te[user] == 0:
            continue
        position = 0
        for i in range(k):
            if I_u_te[user][top_k_indices[user][i]] == 1:
                position = i + 1
                break
        if position:
            RR_u += 1 / position
    return RR_u / U_te_num


def ca_test(r_ui_predict, y_ui_test, k, U_te, U_te_num):
    print("%.4f" % Pre_k(r_ui_predict, y_ui_test, k, U_te, U_te_num)[0])
    print("%.4f" % Rec_k(r_ui_predict, y_ui_test, k, U_te, U_te_num)[0])
    print("%.4f" % F1_k(r_ui_predict, y_ui_test, k, U_te, U_te_num))
    print("%.4f" % NDCG_k(r_ui_predict, y_ui_test, k, U_te, U_te_num))
    print("%.4f" % call_k(r_ui_predict, y_ui_test, k, U_te, U_te_num))
    print("%.4f" % MRR(r_ui_predict, y_ui_test, k, U_te, U_te_num))


if __name__ == "__main__":
    ca_predict()
    # 计算有喜欢项目的用户数
    U_te = np.zeros(n)
    U_te_num = 0
    for i in range(n):
        if np.sum(y_ui_test[i]):
            U_te[i] = 1
            U_te_num += 1
    print("%.4f" % Pre_k(r_ui_predict, y_ui_test, 5 , U_te, U_te_num)[0])
    print("%.4f" % Rec_k(r_ui_predict, y_ui_test, 5, U_te, U_te_num)[0])
    print("%.4f" % F1_k(r_ui_predict, y_ui_test, 5, U_te, U_te_num))
    print("%.4f" % NDCG_k(r_ui_predict, y_ui_test, 5, U_te, U_te_num))
    print("%.4f" % call_k(r_ui_predict, y_ui_test, 5, U_te, U_te_num))
    print("%.4f" % MRR(r_ui_predict, y_ui_test, 5, U_te, U_te_num))
