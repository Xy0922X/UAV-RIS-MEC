from agents import Agent, UE, BS, UAV, Building, IRS
import math as mt
import math
import numpy as np

#
M = 100
phase_shift = 100
B = 8000  # 8MHz
Xi = mt.pow(10, (3 / 10))
white_noise = 10
Power_ue = 3 * mt.pow(10, 3)  # 300mW
Power_uav = 5 * mt.pow(10, 3)  # 500mW
N_0 = mt.pow(10, (-169 / 10)) * (0.1 * mt.pow(10, 3)) / B
omega1 = 0.5
omega2 = 0.5
chi = 1e-26

ue1 = UE.UE(1400, 1650, 0)
ue2 = UE.UE(1700, 1700, 0)
ue3 = UE.UE(2000, 1500, 0)
ue4 = UE.UE(3800, 3500, 0)
ue5 = UE.UE(4200, 2000, 0)

uav1 = UAV.UAV(1000, 2000, 10, 1500000)  # 根据参考论文，无人机初始能量为 500kJ，此处定为 1500kJ 是为了让仿真时无人机可以飞得久一点，到 90 多秒再降落
irs1 = IRS.IRS(1, 1800, 60)

bs1 = BS.BS_local(1600, 1750, 20)
bs2 = BS.BS(3400, 3200, 20)
bs3 = BS.BS(3800, 3400, 20)
bs4 = BS.BS(4400, 2100, 20)
bs5 = BS.BS(4800, 1600, 20)

building1 = Building.Building(1000, 1500, 0, 260, 140, 20)
building2 = Building.Building(1000, 1800, 0, 260, 140, 20)
building3 = Building.Building(1300, 1400, 0, 260, 140, 20)
building4 = Building.Building(1300, 1800, 0, 260, 140, 20)
building5 = Building.Building(1600, 1400, 0, 260, 140, 20)
building6 = Building.Building(1800, 1800, 0, 260, 140, 20)
building7 = Building.Building(3000, 3000, 0, 260, 140, 20)
building8 = Building.Building(3000, 3300, 0, 260, 140, 20)
building9 = Building.Building(3000, 3600, 0, 260, 140, 20)
building10 = Building.Building(3600, 3000, 0, 260, 140, 20)
building11 = Building.Building(3600, 3300, 0, 260, 140, 20)
building12 = Building.Building(3600, 3600, 0, 260, 140, 20)
building13 = Building.Building(4200, 1800, 0, 260, 140, 20)
building14 = Building.Building(4600, 2000, 0, 260, 140, 20)
building15 = Building.Building(4600, 2500, 0, 260, 140, 20)

irs_cluster = [irs1]
ue_cluster = [ue1, ue2, ue3, ue4, ue5]
uav_cluster = [uav1]
bs_cluster = [bs1, bs2, bs3, bs4, bs5]
building_cluster = [building1, building2, building3, building4, building5, building6, building7, building8,
                    building9, building10, building11, building12, building13, building14, building15]


# 根据已知两点坐标，求过这两点的直线解析方程： a1*x+b1*y+c1 = 0  (a >= 0)  &  a2*x+b2*z+c2 = 0  (a >= 0)
def get_linear_equation(ue, uav):
    [p1x, p1y, p1z, p2x, p2y, p2z] = [ue.x, ue.y, ue.z, uav.x, uav.y, uav.z]
    sign1 = 1
    a1 = p2y - p1y
    if a1 < 0:
        sign1 = -1
        a1 = sign1 * a1
    b1 = sign1 * (p1x - p2x)
    c1 = sign1 * (p1y * p2x - p1x * p2y)
    sign2 = 1
    a2 = p2z - p1z
    if a2 < 0:
        sign2 = -1
        a2 = sign2 * a2
    b2 = sign2 * (p1x - p2x)
    c2 = sign2 * (p1z * p2x - p1x * p2z)
    return [a1, b1, c1, a2, b2, c2]


#   视距判断
def line_of_sight_judgement(ue_cluster, uav_cluster, building_cluster):
    los_judgement = np.ones((len(ue_cluster), len(uav_cluster)))
    rows_index = -1
    for rows in los_judgement:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            coefficients = get_linear_equation(ue_cluster[rows_index], uav_cluster[cols_index])
            for x_sample in np.arange(ue_cluster[rows_index].x, uav_cluster[cols_index].x, 0.1):
                y_sample = (- coefficients[2] - coefficients[0] * x_sample) / coefficients[1]
                z_sample = (- coefficients[5] - coefficients[3] * x_sample) / coefficients[4]
                for building in building_cluster:
                    if building.x <= x_sample <= (building.x + building.dx) and \
                            building.y <= y_sample <= (building.y + building.dy) and \
                            building.z <= z_sample <= (building.z + building.dz):
                        los_judgement[rows_index][cols_index] = 0
    return los_judgement


# 智能体间的距离
def get_clusters_distance(agent_cluster1,
                          agent_cluster2):  # https://blog.csdn.net/Tan_HandSome/article/details/82501902
    agent_cluster1_locations = []
    agent_cluster2_locations = []
    for agent1 in agent_cluster1:
        agent_cluster1_locations.append(agent1.xyz)
    for agent2 in agent_cluster2:
        agent_cluster2_locations.append(agent2.xyz)
    A = np.array(agent_cluster1_locations)
    B = np.array(agent_cluster2_locations)
    BT = B.transpose()
    vecProduct = np.dot(A, BT)  # dot production
    SqA = A ** 2  # square of every element in A
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProduct.shape[1]))
    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProduct.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProduct
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    distance = np.array(ED)
    return distance


# GU-UAV之间的传输速率
def get_gu_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster, irs_cluster):
    ue_uav_communicate_rate = np.zeros((len(ue_cluster), len(uav_cluster)))
    los_judgement = line_of_sight_judgement(ue_cluster, uav_cluster, building_cluster)
    rows_index = -1
    ue_uav_distance = get_clusters_distance(ue_cluster, uav_cluster)
    ue_irs_distance = get_clusters_distance(ue_cluster, irs_cluster)
    irs_uav_distance = get_clusters_distance(irs_cluster, uav_cluster)
    gu_uav_communication = np.zeros((len(ue_cluster), len(uav_cluster)))

    for rows in los_judgement:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            if cols == 0:  # 使用智能反射面
                phase_shift = 0
                for m in range(M):
                    angle_dif = np.abs(
                        0.5 * m * (irs_cluster[0].x - ue_cluster[rows_index].x) / ue_irs_distance[rows_index][
                            cols_index] - (irs_cluster[0].x - uav_cluster[cols_index].x) / irs_uav_distance[0][0] * (
                                2 * np.pi))
                    phase_shift = phase_shift + mt.sin((angle_dif / 180) * mt.pi)
                gu_uav_communication[rows_index][cols_index] = Power_ue * math.pow(
                    phase_shift * Xi / (B * N_0 * irs_uav_distance * ue_irs_distance[rows_index][cols_index]),
                    2)
            elif cols == 1:
                gu_uav_communication[rows_index][cols_index] = Power_ue * math.pow(
                    Xi / (B * N_0 * ue_uav_distance[rows_index][cols_index]),
                    2)
            ue_uav_communicate_rate[rows_index][cols_index] = B * np.log2(
                1 + gu_uav_communication[rows_index][cols_index])
    return ue_uav_communicate_rate


def get_ue_to_uav_path_loss(ue_cluster, uav_cluster, building_cluster):
    fc = 2e9  # 单位 HZ
    velocity_c = 3e8  # 光速，单位 m/s
    ue_uav_distance = get_clusters_distance(ue_cluster, uav_cluster)
    path_loss = 20 * np.log10(4 * math.pi * fc * ue_uav_distance / velocity_c)
    los_judgement = line_of_sight_judgement(ue_cluster, uav_cluster, building_cluster)
    rows_index = -1
    for rows in los_judgement:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            if cols == 0:
                path_loss[rows_index][cols_index] = path_loss[rows_index][cols_index] + 5
            elif cols == 1:
                path_loss[rows_index][cols_index] = path_loss[rows_index][cols_index] + 1
    return path_loss


def get_ue_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster):
    ue_uav_communicate_rate = np.zeros((len(ue_cluster), len(uav_cluster)))
    ue_uav_received_power = np.zeros((len(ue_cluster), len(uav_cluster)))
    sinr = np.zeros((len(ue_cluster), len(uav_cluster)))
    ue_uav_path_loss = get_ue_to_uav_path_loss(ue_cluster, uav_cluster, building_cluster)
    # SINR 根据 ADVISOR-002 论文中的公式（8）、（9）建模计算
    rows_index = -1
    for rows in ue_uav_path_loss:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            path_loss_watt = 10 ** (ue_uav_path_loss[rows_index][cols_index] / 20)
            ue_uav_received_power[rows_index][cols_index] = ue_cluster[rows_index].S / path_loss_watt
    # 计算 ue-uav 的干扰 sinr 时需要考虑连接关系（卸载策略），计算某一特定用户所受干扰时，使用的 path-loss 及信道模型以该特定用户的连接关系（即该用户是与无人机还是与基站连接）决定；计算其它 sinr 同理
    sum_ue_uav_received_power = ue_uav_received_power.sum(axis=0)
    white_noise = 10
    rows_index = -1
    for rows in ue_uav_received_power:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            sinr[rows_index][cols_index] = cols / (sum_ue_uav_received_power[cols_index] - cols + white_noise ** 2)
            ue_uav_communicate_rate[rows_index][cols_index] = ue_cluster[rows_index].bandwidth * math.log2(
                1 + sinr[rows_index][cols_index])
    return ue_uav_communicate_rate


# print(line_of_sight_judgement(ue_cluster, uav_cluster, building_cluster))
# print(get_ue_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster))
# print(get_gu_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster, irs_cluster))
# print(1e6 / get_gu_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster, irs_cluster))
# print(1e6 / get_ue_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster))

# agent_cluster 上的任务执行时间
def get_Tc_agent_cluster(agent_cluster):
    Tc_agent_cluster = []
    rows_index = -1
    for rows in agent_cluster:
        rows_index += 1
        Tc_agent_cluster.append(rows.cpu_task / rows.cpu_power)  # 执行时间
    return Tc_agent_cluster


# uav 到 bs 的路径损耗的计算
# 根据 ADVISOR-006 论文中的公式（12）建模计算
def get_uav_to_bs_path_loss(uav_cluster, bs_cluster):
    uav_cluster_locations = []
    for uav in uav_cluster:
        uav_cluster_locations.append(uav.xyz)
    uav_cluster_locations = np.array(uav_cluster_locations)
    h_uav = uav_cluster_locations[:, 2]
    fc = 0.25e3  # 单位 MHZ
    d = get_clusters_distance(uav_cluster, bs_cluster)
    x = np.tile(np.maximum(23.9 - 1.8 * np.log10(h_uav), 20), (len(bs_cluster), 1)).transpose()
    path_loss = x * np.log10(d) + 20 * math.log10(40 * math.pi * fc / 3) - 60
    return path_loss


def get_uav_bs_communicate_rate(uav_cluster, bs_cluster):
    uav_bs_communicate_rate = np.zeros((len(uav_cluster), len(bs_cluster)))
    uav_bs_received_power = np.zeros((len(uav_cluster), len(bs_cluster)))
    sinr = np.zeros((len(uav_cluster), len(bs_cluster)))
    uav_bs_path_loss = get_uav_to_bs_path_loss(uav_cluster, bs_cluster)
    rows_index = -1
    for rows in uav_bs_path_loss:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            path_loss_watt = 10 ** (uav_bs_path_loss[rows_index][cols_index] / 20)
            uav_bs_received_power[rows_index][cols_index] = uav_cluster[rows_index].S / path_loss_watt
    sum_uav_bs_received_power = uav_bs_received_power.sum(axis=0)
    white_noise = 10
    rows_index = -1
    for rows in uav_bs_received_power:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            sinr[rows_index][cols_index] = cols / (sum_uav_bs_received_power[cols_index] - cols + white_noise ** 2)
            uav_bs_communicate_rate[rows_index][cols_index] = uav_cluster[rows_index].bandwidth * math.log2(
                1 + sinr[rows_index][cols_index])
    return uav_bs_communicate_rate


# print(1e6 / get_uav_bs_communicate_rate(uav_cluster, bs_cluster))


def get_uav_bs_cluster_offload_strategy_and_T(uav_cluster, bs_cluster, cpu_task, data_size):
    # 基站处理的任务量
    bs_cluster_task = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机与基站的卸载策略
    uav_bs_cluster_offload_strategy = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机总能耗
    E_uav = 0
    # 总时延
    T_uav_bs_cluster = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机本地处理的时延
    T_uav_de = 0
    # 无人机本地处理的能耗
    E_uav_de = 0
    # 无人机传输能耗
    E_uav_tr = 0
    # 无人机传输时延 + 基站处理时延
    T_uav_tr_bs_cluster = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机与基站之间的传输速率
    uav_bs_clusters_communicate_rate = get_uav_bs_communicate_rate(uav_cluster, bs_cluster)
    rows_index = -1
    # 对于每个无人机
    for rows in uav_cluster:
        rows_index += 1
        # 传输时延
        T_uav_tr_bs_cluster += data_size / np.array(uav_bs_clusters_communicate_rate)
        # 计算量加到基站上
        for i in range(len(bs_cluster)):
            bs_cluster[i].cpu_task += cpu_task
        # 计算时延
        T_uav_tr_bs_cluster += np.array(get_Tc_agent_cluster(bs_cluster))
        T_uav_de += cpu_task / uav_cluster[rows_index].cpu_power
        # 计算能耗
        E_uav_de += chi * (cpu_task) ** 2 * uav_cluster[rows_index].cpu_power * 100
        # 传输能耗
        E_uav_tr += Power_uav * data_size / (np.array(uav_bs_clusters_communicate_rate) * 10)
        # return T_uav_tr_bs_cluster, T_uav_de, E_uav_de, E_uav_tr
        if T_uav_de < np.min(T_uav_tr_bs_cluster):
            T_uav_bs_cluster = T_uav_de
            E_uav += E_uav_de
        else:
            T_uav_bs_cluster = np.min(T_uav_tr_bs_cluster)
            min_delay_bs_index = np.argmin(T_uav_tr_bs_cluster, axis=None)
            min_list = np.unravel_index(min_delay_bs_index, T_uav_tr_bs_cluster.shape, order='C')
            # 最小值的位置为
            min_location = min_list[1]
            uav_bs_cluster_offload_strategy[0][min_location] = 1.0
            bs_cluster_task[0][min_location] += cpu_task
            E_uav += E_uav_tr[0][min_location]
    return uav_bs_cluster_offload_strategy, T_uav_bs_cluster, bs_cluster_task, E_uav


# print(get_uav_bs_cluster_offload_strategy_and_T(uav_cluster, bs_cluster, 3e9, 3e6))


def get_strategy_by_action_for_maddpg(ue_cluster, uav_cluster, bs_cluster, building_cluster, irs_cluster, action,
                                      ue_index):
    #  action 为0.25 或者 0.75
    # 基站处理的任务量
    bs_cluster_task = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机与基站的卸载策略
    uav_bs_cluster_offload_strategy = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机总能耗
    E_uav = 0
    ue_uav_cluster_offload_strategy = np.zeros((len(uav_cluster)))
    #  ue_uav端的总时延
    T_ue_uav_cluster = np.zeros(len(uav_cluster))
    ue_uav_clusters_communicate_rate = get_gu_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster,
                                                                   irs_cluster)
    division_num_of_one_ue = 1 + len(uav_cluster)  # 维度为 无人机个数 + 用户
    division = 1 / division_num_of_one_ue  # 0.5
    if action[0] <= 0 + division * 1:
        ue_local_strategy = 1  # UE本地处理置1
        T_ue_uav_cluster = 1e9 / 1e8
    elif action[0] <= 0 + division * (1 + len(uav_cluster)):
        #  没考虑选择哪个无人机，场景目前只有一个无人机
        for j in range(0, len(uav_cluster)):
            #  UAV_j与UE的连接置1
            ue_uav_cluster_offload_strategy[j] += 1
            # 计算UE-UAV的传输时延
            T_ue_uav_cluster[j] += 1e6 / ue_uav_clusters_communicate_rate[ue_index][j]
            # UAV的任务计算量 += UE的任务计算量
            uav_cluster[j].cpu_task += 1e9
            # UAV的任务量 += UE的任务量
            uav_cluster[j].data_size += 1e6
            # UE的任务量和计算量置0
            ue_cluster[ue_index].cpu_task -= 1e9
            ue_cluster[ue_index].data_size -= 1e6
            uav_bs_cluster_offload_strategy, T_uav_bs_cluster, bs_cluster_task_, E_uav_ = get_uav_bs_cluster_offload_strategy_and_T(
                uav_cluster, bs_cluster, uav_cluster[j].cpu_task, uav_cluster[j].data_size)
            uav_cluster[j].cpu_task -= uav_cluster[j].cpu_task
            uav_cluster[j].data_size -= uav_cluster[j].data_size
            T_ue_uav_cluster[j] += T_uav_bs_cluster
            bs_cluster_task += bs_cluster_task_
            E_uav += E_uav_
    # 时延，基站处理的任务量
    return uav_bs_cluster_offload_strategy, T_ue_uav_cluster, bs_cluster_task, E_uav


print(get_strategy_by_action_for_maddpg(ue_cluster, uav_cluster, bs_cluster, building_cluster, irs_cluster,
                                        [0.25], 0))


# bs_task_sum = np.zeros((1, len(bs_cluster)))
# bs_task_sum += \
# get_strategy_by_action_for_maddpg(ue_cluster, uav_cluster, bs_cluster, building_cluster, irs_cluster, [0.9, 0.2, 0.4, 0.8, 0.8])[1]
#
#
# ratio_task = np.zeros((1, len(bs_cluster)))
# for i in range(len(bs_cluster)):
#     ratio_task[0][i] += bs_task_sum[0][i] / np.sum(bs_task_sum)
#
# # print(ratio_task)
# # print(len(ratio_task[0]))
# numerator = 0
# for i in range(len(ratio_task[0])):
#     numerator += math.pow(ratio_task[0][i], 2)
# numerator *= len(ratio_task[0])

# print(numerator)

def get_power_cost_per_meter(velocity):
    return 580.65 * (1 + 0.000075 * velocity ** 2) + \
           790.67 * ((1 + velocity ** 4 / (103.68 ** 2)) ** (1 / 2) - velocity ** 2 / 103.68) ** (1 / 2) + \
           0.00726 * velocity ** 3
