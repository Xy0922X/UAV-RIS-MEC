import random
import math

# 初始化基站和无人机的位置
base_stations = [(0, 0, 0), (10, 10, 10), (20, 20, 20)]
drone = (5, 5, 5)

# 初始化每个基站的时延和能耗值
base_station_values = [{'latency': 5, 'energy': 10}, {'latency': 10, 'energy': 20}, {'latency': 15, 'energy': 30}]

# 定义一个函数，用于计算无人机连接基站的任务时延和能耗比
def calculate_score(station):
    latency = base_station_values[station]['latency']
    energy = base_station_values[station]['energy']
    score = latency / energy
    return score

# 初始化当前最优解的基站和对应的分数
current_station = 0
current_score = calculate_score(current_station)

# 定义登山算法的迭代次数
iterations = 1000

# 定义每次迭代中的最大步长
max_step = 5

# 定义循环，用于迭代计算
for i in range(iterations):
    # 随机选择一个步长，并且选择一个基站的邻居作为新的候选解
    step = random.uniform(0, max_step)
    neighbor_station = random.randint(0, len(base_stations)-1)
    neighbor = tuple([a + step * (b - a) for a, b in zip(drone, base_stations[neighbor_station])])

    # 计算新的候选解的得分
    neighbor_score = calculate_score(neighbor_station)

    # 如果新的候选解的得分优于当前最优解的得分，更新当前最优解
    if neighbor_score < current_score:
        current_station = neighbor_station
        current_score = neighbor_score

# 输出结果
print("无人机连接的基站为：", current_station)
print("基站的坐标为：", base_stations[current_station])
print("基站的得分为：", current_score)
