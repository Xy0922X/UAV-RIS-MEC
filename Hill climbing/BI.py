from itertools import combinations
import random as r
import numpy as np

# 计算目标函数:距离
def get_tour_length(sol, distance):
    tour = sol.copy()
    n = len(tour)
    length = 0
    tour.append(tour[0])
    for k in range(n):
        i = tour[k]
        j = tour[k + 1]
        length += distance[i, j]
    return length

# 计算距离矩阵
def getDistance(x, y):
    n = len(x)
    distance = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i, n):
            distance[i, j] = np.sqrt(pow((x[i] - x[j]), 2) + pow((y[i] - y[j]), 2))
    distance += distance.T - np.diag(distance.diagonal())
    return distance

# 请在此处添加代码，实现目标函数功能
#********** Begin **********#
def getActionlist(n):
    list1 = []
    for m1 in range(n - 1):
        for n1 in range(m1 + 1, n):
            list1.append([m1, n1])

    action_list = list1
    return action_list


def getBestImprovement(sol, action_list, distance):
    length_init = get_tour_length(sol, distance)
    num = len(action_list)

    sol_text = sol
    while True:
        length_init1 = length_init
        for n2 in range(num):
            p = action_list[n2]
            a = sol_text[p[0]]
            b = sol_text[p[1]]
            sol_text[p[1]] = b
            sol_text[p[0]] = a
            length = get_tour_length(sol_text, distance)
            if length < length_init:
               length_init = length
               sol = sol_text
            else:
                sol_text[p[0]] = a
                sol_text[p[1]] = b



        if length_init1 == length_init:
             break

    return sol

#*********** End ***********#

x = [82, 91, 12, 92, 63, 9, 28, 55, 96, 97, 15, 98]  # 城市坐标
y = [14, 42, 92, 80, 96, 66, 3, 85, 94, 68, 76, 75]

sol = [11, 9, 10, 5, 2, 6, 0, 1, 7, 4, 8, 3]  # 当前解
soll = [4, 7, 2, 10, 5, 6, 0, 1, 9, 11, 3, 8]  # 当前解
distance = getDistance(x, y)
length_init = get_tour_length(sol, distance)
print(length_init)
n = len(x)
# 测试函数
action_list = getActionlist(n)  # 格式为[(x1,x2),(x1,x3),...]
new_sol = getBestImprovement(sol, action_list, distance)

# 输出结果
print(new_sol)

