import math

import numpy as np

user_num = 943
item_num = 1682

user_id = [i for i in range(1, 944)]
item_id = [i for i in range(1, 1683)]

rm = np.zeros((user_num, item_num), int)  # rating matrix
y_ui = np.zeros((user_num, item_num), int)

with open("D:\Recommendation\learning_plan\slides_2\AF\ml-100k\\u1.base", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        rm[int(user) - 1][int(item) - 1] = int(rating)
        if rating:
            y_ui[int(user) - 1][int(item) - 1] = 1

# number of observed ratings  p = |R|
p = np.sum(y_ui)

density = p / user_num / item_num

# global average rating
r_ = rm.sum() / p
# average rating of user u
r1 = rm.sum(axis=0)
r2 = np.sum(y_ui, axis=0)  # 每列的非0元素个数
ru_ = np.zeros(item_num)
for i in range(item_num):
    if r2[i] == 0:
        ru_[i] = r_  # 默认值处理
    else:
        ru_[i] = r1[i] / r2[i]

# average rating of item i
r3 = rm.sum(axis=1)
r4 = np.sum(y_ui, axis=1)  # 每行的非0元素个数
ri_ = np.zeros(user_num)
for i in range(user_num):
    if r4[i] == 0:
        ri_[i] = r_  # 默认值处理
    else:
        ri_[i] = r3[i] / r4[i]
# bias of user u
bu = np.zeros(item_num)
for i in range(item_num):
    if r2[i] == 0:
        bu[i] = 0
    else:
        bu[i] = np.sum(y_ui[:, i] * (rm[:, i] - ri_)) / r2[i]
# bias of item i
bi = np.zeros(user_num)
for i in range(user_num):
    if r4[i] == 0:
        bi[i] = 0
    else:
        bi[i] = np.sum(y_ui[i, :] * (rm[i, :] - ru_)) / r4[i]


def MAE(rui, rui_, r):
    return np.sum(np.abs(rui - rui_)) / r


def RMSE(rui, rui_, r):
    return math.sqrt(np.sum((rui - rui_) * (rui - rui_)) / r)


# Predicted rating of user u on item i
rm_ = np.zeros((user_num, item_num), int)
with open("D:\Recommendation\learning_plan\slides_2\AF\ml-100k\\u1.test", 'r') as k:
    for line in k.readlines():
        user, item, rating, _ = line.split('	')
        rm_[int(user) - 1][int(item) - 1] = int(rating)
rm_1 = np.zeros((user_num, item_num), float)
for i in range(user_num):
    for j in range(item_num):
        if rm_[i][j]:
            rm_1[i][j] = ru_[j]

rm_2 = np.zeros((user_num, item_num), float)
for i in range(user_num):
    for j in range(item_num):
        if rm_[i][j]:
            rm_2[i][j] = ri_[i]

rm_3 = np.zeros((user_num, item_num), float)
for i in range(user_num):
    for j in range(item_num):
        if rm_[i][j]:
            rm_3[i][j] = ri_[i] / 2 + ru_[j] / 2

rm_4 = np.zeros((user_num, item_num), float)
for i in range(user_num):
    for j in range(item_num):
        if rm_[i][j]:
            rm_4[i][j] = bu[j] + ri_[i]

rm_5 = np.zeros((user_num, item_num), float)
for i in range(user_num):
    for j in range(item_num):
        if rm_[i][j]:
            rm_5[i][j] = bi[i] + ru_[j]

rm_6 = np.zeros((user_num, item_num), float)
for i in range(user_num):
    for j in range(item_num):
        if rm_[i][j]:
            rm_6[i][j] = bi[i] + bu[j] + r_
r = np.count_nonzero(rm_)
print(f"{RMSE(rm_, rm_1, r):.4f}, {RMSE(rm_, rm_2, r):.4f}, {RMSE(rm_, rm_3, r):.4f}, "
      f"{RMSE(rm_, rm_4, r):.4f}, {RMSE(rm_, rm_5, r):.4f},{RMSE(rm_, rm_6, r):.4f}")
print(f"{MAE(rm_, rm_1, r):.4f}, {MAE(rm_, rm_2, r):.4f}, {MAE(rm_, rm_3, r):.4f}, "
      f"{MAE(rm_, rm_4, r):.4f}, {MAE(rm_, rm_5, r):.4f}, {MAE(rm_, rm_6, r):.4f}")
