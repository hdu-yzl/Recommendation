import random

import numpy as np
import math

user_num = 943
item_num = 1682

K = 50
rm = np.zeros((user_num, item_num), int)  # rating matrix
y_ui = np.zeros((user_num, item_num), int)


def MAE(rui, rui_, r):
    return np.sum(np.abs(rui - rui_)) / r
def RMSE(rui, rui_, r):
    return math.sqrt(np.sum((rui - rui_) * (rui - rui_)) / r)


with open("D:\Recommendation\learning_plan\slides_2\AF\ml-100k\\u1.base", 'r') as k:
    for line in k.readlines():
        user, item, rating, z = line.split('	')
        rm[int(user) - 1][int(item) - 1] = int(rating)
        if rating:
            y_ui[int(user) - 1][int(item) - 1] = 1

# number of observed ratings  p = |R|
p = np.sum(y_ui)

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


def SM(u1_num):
    u1_y = y_ui[u1_num]
    u1_y = np.repeat(u1_y,user_num,axis=0)
    u1_y = u1_y*y_ui  # w 和 u 的共有元素矩阵
    rm_u = ru_.repeat(len(ru_)).reshape(-1, len(ru_)) #把ru_每项扩展为一行
    rm_u1 = (rm[u1_num]-ru_[u1_num]).repeat(user_num).reshape(-1, user_num)
    rm_uk = (rm - rm_u)*rm_u1*u1_y

    swu_1 =np.sum(rm_uk,axis=1)
    swu_2 = np.sqrt(np.sum((rm-rm_uk)*(rm-rm_uk)*u1_y,axis=1))
    swu_3 = np.sqrt(np.sum((rm_u1-rm_uk)*(rm_u1-rm_uk),axis=1))

    swu = np.where(swu_2 != 0, swu_1 / swu_2, np.nan)  # 忽略 swu_2 中的0
    swu = np.where(swu_3 != 0, swu / swu_3, np.nan)
    swu[swu==0]=-1
    swu = np.nan_to_num(swu, nan=-1)
    return swu , (rm-rm_u)*u1_y


def pr_r(u,rm_p_y):  # 计算预测值
    r_uj_ = 0
    r_uj_ += ru_[u]
    swu,rm_s = SM(u)
    r_u_j = np.zeros(item_num,float)


    return

rm_ = np.zeros((user_num, item_num), int)
rm_p_y= np.zeros((user_num, item_num), int)
rm_p = np.zeros((user_num, item_num), int)  # 预测值
with open("D:\Recommendation\learning_plan\slides_2\AF\ml-100k\\u1.test", 'r') as k:
    for line in k.readlines():
        user, item, rating, _ = line.split('	')
        rm_[int(user) - 1][int(item) - 1] = int(rating)
        rm_p_y[int(user) - 1][int(item) - 1] = 1


for i in range(user_num):
    rm_p[i]= pr_r(i,rm_p_y)

r = np.count_nonzero(rm_)
print(f"{RMSE(rm_, rm_p, r):.4f}")
print(f"{MAE(rm_, rm_p, r):.4f}")
