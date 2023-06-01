import math
import numpy as np
import math as mt

M = 100
phase_shift = 100
B = 8000  # 8MHz
Xi = mt.pow(10, (3 / 10))
white_noise = 10
Power_ue = 3 * mt.pow(10, 3)  # 300mW
Power_uav = 5 * mt.pow(10, 3)  # 500mW
N_0 = mt.pow(10, (-169 / 10)) * (0.1 * mt.pow(10, 3)) / B
omega1 = 0.05
omega2 = 1
chi = 1e-26

# 起飞
def launching(self):
    uav_index = -1
    for uav in self.uav_cluster:
        uav_index += 1
        # 让无人机高度抬高90m
        for _ in np.arange(1, 21, 1):
            theta = 0  # 单位时间内的飞行仰角（ 0 表示垂直起飞）
            fai = 0  # 单位时间内飞行轨迹与x轴正方向的水平夹角，垂直起降时 fai 取值随意
            velocity = 3  # 飞行速度，m/s
            uav.energy = uav.energy - get_power_cost_per_meter(abs(velocity)) * abs(velocity) * 1
            uav.flight_trace(theta, fai, velocity, 1)  # 直线飞行
            self.uav_cluster_x[uav_index].append(uav.x)
            self.uav_cluster_y[uav_index].append(uav.y)
            self.uav_cluster_z[uav_index].append(uav.z)
    return self.uav_cluster


# 降落
def landing(self, uav, uav_index):
    for _ in np.arange(1, 3000, 1):
        theta = math.pi  # 单位时间内的飞行仰角（ pi 表示垂直降落）
        fai = 0  # 单位时间内飞行轨迹与x轴正方向的水平夹角，垂直起降时 fai 取值随意
        velocity = 3  # 飞行速度，m/s
        if uav.z >= 1 and (uav.z - velocity) < 1:
            velocity = uav.z - 1
        uav.energy = uav.energy - get_power_cost_per_meter(abs(velocity)) * abs(velocity) * 1
        uav.flight_trace(theta, fai, velocity, 1)  # 直线飞行

        self.uav_cluster_x[uav_index].append(uav.x)
        self.uav_cluster_y[uav_index].append(uav.y)
        self.uav_cluster_z[uav_index].append(uav.z)

        if uav.z == 1:
            break


# 得到多点连线形成的多边形的中心点
# 按该方法得到的中心点物理意义不明，已弃用
# reference - https://www.csdn.net/tags/NtzaQg2sNjUxOTUtYmxvZwO0O0OO0O0O.html
def get_center_location(clusters_points):
    x = 0  # lon
    y = 0  # lat
    z = 0
    length = len(clusters_points)
    for lon, lat in clusters_points:
        lon = math.radians(float(lon))
        #  radians(float(lon))   Convert angle x from degrees to radians
        # 	                    把角度 x 从度数转化为 弧度
        lat = math.radians(float(lat))
        x += math.cos(lat) * math.cos(lon)
        y += math.cos(lat) * math.sin(lon)
        z += math.sin(lat)
        x = float(x / length)
        y = float(y / length)
        z = float(z / length)
    return math.degrees(math.atan2(y, x)), math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))


# Gauss's area formula 高斯面积计算，多边形形心计算中使用
def cal_area(vertices):  # Gauss's area formula 高斯面积计算
    A = 0.0
    point_p = vertices[-1]
    for point in vertices:
        A += (point[1]*point_p[0] - point[0]*point_p[1])
        point_p = point
    return abs(A)/2


# 得到多点连线形成的多边形的形心，需要逆时针输入各点坐标，输入格式：[[-1., -1.], [-2., -1.], [-2., -2.], [-1., -2.]]
# reference - https://blog.csdn.net/kindlekin/article/details/121318530
def cal_centroid(points):
    A = cal_area(points)
    c_x, c_y = 0.0, 0.0
    point_p = points[-1]  # point_p 表示前一节点
    for point in points:
        c_x += ((point[0] + point_p[0]) * (point[1]*point_p[0] - point_p[1]*point[0]))
        c_y += ((point[1] + point_p[1]) * (point[1]*point_p[0] - point_p[1]*point[0]))
        point_p = point

    return c_x / (6*A), c_y / (6*A)


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


# 视距判断
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


# 坠机判断（无人机与建筑物相撞或无人机之间相撞）
def uav_crash_judgement(uav_cluster, building_cluster):
    crash_flag1 = False
    uav_uav_distance = get_clusters_distance(uav_cluster, uav_cluster)
    n = len(uav_cluster)
    # 无人机不会与其自身碰撞，直接给一个较大的距离
    for i in range(n):
        uav_uav_distance[i][i] = 1000
    for uav in uav_cluster:
        for building in building_cluster:
            if building.x <= uav.x <= (building.x + building.dx) and \
                    building.y <= uav.y <= (building.y + building.dy) and \
                    building.z <= uav.z <= (building.z + building.dz):
                crash_flag1 = True
    crash_flag2 = (uav_uav_distance <= 1).any()
    crash_judgement = crash_flag1 or crash_flag2
    return crash_judgement


# agent_cluster1 与 agent_cluster2 之间的距离
def get_clusters_distance(agent_cluster1, agent_cluster2):
    #  https://blog.csdn.net/Tan_HandSome/article/details/82501902
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


# ue 到 uav 的路径损耗的计算
# 根据 ADVISOR-007 论文中的公式（1）—（6）建模计算
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


# ue 到 bs 的路径损耗的计算
# 根据 ADVISOR-006 论文中的公式（5）—（8）建模计算
def get_ue_to_bs_path_loss(ue_cluster, bs_cluster):
    fc = 2e3  # 单位 MHZ
    hbs = 50  # 单位 m，计划放在房顶，所以比论文里的 30m 高一点
    hue = 2  # 单位 m
    d = get_clusters_distance(ue_cluster, bs_cluster)
    a_hue = (1.1*np.log10(fc) - 0.7) * hue - 1.56 * np.log10(fc) - 0.8
    A = 69.55 + 26.16 * np.log10(fc) - 13.82 * np.log10(hbs) - a_hue
    B = 44.9 - 6.55 * np.log10(hbs)
    C = -2 * np.log10(fc / 28) ** 2 - 5.4
    path_loss = A + B * np.log10(d) + C - 60
    return path_loss


# ue 到 bs 的路径损耗的计算2
# 根据 ADVISOR-008 论文中的公式（7），以及参考文献008-[28]中的公式（1）及其相关说明建模计算
# 由于该方法中包含一高斯分布的随机数，会导致计算得到的 path_loss 可能低至 70+ dB，高至 120+ dB（此类极端情况的出现为小概率事件，大部分情况为 98 dB 左右）
# 进一步地，由于该高斯分布随机数大小不定，使计算得到的 path_loss、ue_bs_communicate_rate、Ttr_ue_uav_and_ue_bs_cluster 等数值大小不定，最后反映为系统整体优化指标（时延）的震荡
# 由于上述原因，可能导致系统整体优化指标（时延）出现很大的数值（很大数值指大于1000等，实际人为定义飞行轨迹的场景下，正常可能在400以下震荡）
def get_ue_to_bs_path_loss2(ue_cluster, bs_cluster):
    wl = 0.15  # 波长，单位 m ；频率 fc 为 2e3 MHZ ，光速 velocity_c = wl * fc
    d0 = 1  # 单位 m
    d = get_clusters_distance(ue_cluster, bs_cluster)
    A = 20 * np.log10(4 * math.pi * d0 / wl)  # 见参考文献008-[28]
    B = 10 * 3 * np.log10(d / d0)
    x = np.random.normal(0, np.random.randint(5, 16, 1), 1)  # shadowing effect，是均值为 0 且标准差在 5 到 16 之间的高斯分布值，单位dB
    # path_loss = A + B + x
    path_loss = A + B
    return path_loss


# uav 到 bs 的路径损耗的计算
# 根据 ADVISOR-006 论文中的公式（12）建模计算
def get_uav_to_bs_path_loss(uav_cluster, bs_cluster):
    uav_cluster_locations = []
    for uav in uav_cluster:
        uav_cluster_locations.append(uav.xyz)
    uav_cluster_locations = np.array(uav_cluster_locations)
    h_uav = uav_cluster_locations[:, 2]
    fc = 1.5e3  # 单位 MHZ
    d = get_clusters_distance(uav_cluster, bs_cluster)
    # xx= np.tile(h_uav,(3,1)).transpose()
    x = np.tile(np.maximum(23.9 - 1.8 * np.log10(h_uav), 20), (len(bs_cluster), 1)).transpose()
    path_loss = x * np.log10(d) + 20 * math.log10(40 * math.pi * fc / 3) - 60
    return path_loss


# 无人机与 uav0 之间的距离
def get_uav_uav0_distance(uav, uav0):
    return ((uav.x - uav0.x) ** 2 + (uav.y - uav0.y) ** 2 + (uav.z - uav0.z) ** 2) ** 0.5


# 无人机与 uav0 之间连线与x轴的水平夹角
# 已弃用
def get_uav_uav0_fai(uav, uav0):
    return np.arctan((uav.y - uav0.y)/(uav.x - uav0.x))


# ue 到 uav 的数据卸载速率，单位bit/s
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
            ue_uav_communicate_rate[rows_index][cols_index] = ue_cluster[rows_index].bandwidth * math.log2(1 + sinr[rows_index][cols_index])
    return ue_uav_communicate_rate


# ue 到 bs 的数据卸载速率，单位bit/s
def get_ue_bs_communicate_rate(ue_cluster, bs_cluster):
    ue_bs_communicate_rate = np.zeros((len(ue_cluster), len(bs_cluster)))
    ue_bs_received_power = np.zeros((len(ue_cluster), len(bs_cluster)))
    sinr = np.zeros((len(ue_cluster), len(bs_cluster)))
    ue_bs_path_loss = get_ue_to_bs_path_loss2(ue_cluster, bs_cluster)
    # print(ue_bs_path_loss)
    # SINR 根据 ADVISOR-002 论文中的公式（8）、（9）建模计算
    rows_index = -1
    for rows in ue_bs_path_loss:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            path_loss_watt = 10 ** (ue_bs_path_loss[rows_index][cols_index] / 20)
            ue_bs_received_power[rows_index][cols_index] = ue_cluster[rows_index].S / path_loss_watt
    sum_ue_bs_received_power = ue_bs_received_power.sum(axis=0)
    white_noise = 10
    rows_index = -1
    for rows in ue_bs_received_power:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            sinr[rows_index][cols_index] = cols / (sum_ue_bs_received_power[cols_index] - cols + white_noise ** 2)
            ue_bs_communicate_rate[rows_index][cols_index] = ue_cluster[rows_index].bandwidth * math.log2(1 + sinr[rows_index][cols_index])
    return ue_bs_communicate_rate


# uav 到 bs 的数据卸载速率，单位bit/s
def get_uav_bs_communicate_rate(uav_cluster, bs_cluster):
    uav_bs_communicate_rate = np.zeros((len(uav_cluster), len(bs_cluster)))
    uav_bs_received_power = np.zeros((len(uav_cluster), len(bs_cluster)))
    sinr = np.zeros((len(uav_cluster), len(bs_cluster)))
    uav_bs_path_loss = get_uav_to_bs_path_loss(uav_cluster, bs_cluster)
    # SINR 根据 ADVISOR-002 论文中的公式（8）、（9）建模计算
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
            uav_bs_communicate_rate[rows_index][cols_index] = uav_cluster[rows_index].bandwidth * math.log2(1 + sinr[rows_index][cols_index])
    return uav_bs_communicate_rate


# agent_cluster1 到 agent_cluster2 的数据卸载速率，单位bit/s
# 根据距离进行的信道建模，已弃用
def get_clusters_communicate_rate(agent_cluster1, agent_cluster2):
    clusters_communicate_rate = [[]]
    clusters_distance = get_clusters_distance(agent_cluster1, agent_cluster2)
    rows_index = -1
    for rows in clusters_distance:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            N = cols ** 2 / 2  # 噪声功率；根据 ADVISOR-001 论文中的公式（10）建模计算，假设ρ为2
            # 信号强度 S 和 path-loss 都是矩阵形式；sinr 也是矩阵形式，干扰部分在 sinr 矩阵里添加
            sinr = agent_cluster1[rows_index].S / N  # 信噪比，假设为 30 dB 量级，换算为无单位比例值即 1000 左右的量级
            communicate_rate = agent_cluster1[rows_index].bandwidth * math.log2(
                1 + sinr)  # 用香农公式计算出数据传输速率，单位bit/s；具体应当根据 ADVISOR-001 论文中的公式（10）建模计算，此处为建模方便使用常见假设值
            clusters_distance[rows_index][cols_index] = communicate_rate
            clusters_communicate_rate = clusters_distance
    return clusters_communicate_rate


# agent_cluster 上的数据卸载时间；由 agent_cluster 和 cluster_communicate_rate 两个参数可以确认是从哪一类 agent 卸载到哪一类 agent
# 版本迭代后，数据卸载时间在 get_strategy_and_Ttr_by_action_for_maddpg() 方法中直接计算，已弃用
def get_Tr_agent_cluster(agent_cluster, cluster_communicate_rate):
    Tr_agent_cluster = np.zeros((cluster_communicate_rate.shape[0], cluster_communicate_rate.shape[1]))
    rows_index = -1
    for rows in cluster_communicate_rate:
        cols_index = -1
        rows_index += 1
        for cols in rows:
            cols_index += 1
            Tr_agent_cluster[rows_index][cols_index] = agent_cluster[rows_index].data_size / cluster_communicate_rate[rows_index][cols_index]  # 卸载时间
    return Tr_agent_cluster


# agent_cluster 上的任务执行时间
def get_Tc_agent_cluster(agent_cluster):
    Tc_agent_cluster = []
    rows_index = -1
    for rows in agent_cluster:
        rows_index += 1
        Tc_agent_cluster.append(rows.cpu_task / rows.cpu_power)  # 执行时间
    return Tc_agent_cluster


"""
下面两个函数分别表示了ue的卸载策略和uav的卸载策略；
值得注意的是，建模时设计了如下规则：
对ue来说，初始化的时候只有ue上有数据量和任务量，当ue上的卸载策略按判断条件确定下来之后，
再将初始化ue的数据量和任务量加到要卸载的uav或bs对象的数据量和任务量上，如果不卸载，就不用加；
对uav来说，考虑到uav的卸载策略时说明计算任务已经卸载到uav上了，当uav上的卸载策略按判断条件确定下来之后，
再将初始化ue的数据量和任务量加到要卸载的bs对象的数据量和任务量上，如果不卸载，就不用加；
有了如上规则，初始化时就只需要给ue配置数据量和任务量，后面每一次获取卸载策略的时候，uav和bs上的数据量和任务量会发生与卸载策略相对应的变化；
不过在卸载策略的函数里，没有写把被卸载对象上的数据量和任务量清零的逻辑，因为考虑到计算优化指标时可能要根据这些量计算卸载时间，因此给予保留；
进一步地，数据量和任务量清零的逻辑在每一次优化指标计算完毕之后进行；
"""


# 此处为用户设备集群上的任务是否进行卸载的卸载策略，当前假定卸载策略仅与距离有关；同时进行了各用户设备 data_size、cpu_task 的卸载变化，以及卸载传输时延 Ttr 的计算
# 版本迭代后 ue 卸载策略直接在 get_strategy_and_Ttr_by_action_for_maddpg() 方法中计算，已弃用
def get_ue_uav_and_ue_bs_cluster_offload_strategy_and_Ttr(ue_cluster, uav_cluster, bs_cluster, building_cluster):
    ue_uav_cluster_offload_strategy = np.zeros((len(ue_cluster), len(uav_cluster)))
    ue_bs_cluster_offload_strategy = np.zeros((len(ue_cluster), len(bs_cluster)))
    Ttr_ue_uav_cluster = np.zeros((len(ue_cluster), len(uav_cluster)))
    Ttr_ue_bs_cluster = np.zeros((len(ue_cluster), len(bs_cluster)))
    ue_uav_cluster_distance = get_clusters_distance(ue_cluster, uav_cluster)
    ue_bs_cluster_distance = get_clusters_distance(ue_cluster, bs_cluster)
    ue_uav_clusters_communicate_rate = get_ue_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster)
    ue_bs_clusters_communicate_rate = get_ue_bs_communicate_rate(ue_cluster, bs_cluster)
    rows_index = -1
    for rows in ue_uav_cluster_distance:
        rows_index += 1
        rows1 = ue_uav_cluster_distance[rows_index]
        ue_uav_min_index = (np.argwhere(rows1 == min(rows1)))[0][0]
        rows2 = ue_bs_cluster_distance[rows_index]
        ue_bs_min_index = (np.argwhere(rows2 == min(rows2)))[0][0]
        if min(rows1) >= 200 and min(rows2) >= 200:
            pass
        elif ue_uav_cluster_distance[rows_index][ue_uav_min_index] < ue_bs_cluster_distance[rows_index][ue_bs_min_index]:
            ue_uav_cluster_offload_strategy[rows_index][ue_uav_min_index] = 1
            Ttr_ue_uav_cluster[rows_index][ue_uav_min_index] = ue_cluster[rows_index].data_size / ue_uav_clusters_communicate_rate[rows_index][ue_uav_min_index]
            uav_cluster[ue_uav_min_index].cpu_task += ue_cluster[rows_index].cpu_task
            uav_cluster[ue_uav_min_index].data_size += ue_cluster[rows_index].data_size
            ue_cluster[rows_index].cpu_task -= ue_cluster[rows_index].cpu_task
            ue_cluster[rows_index].data_size -= ue_cluster[rows_index].data_size
        else:
            ue_bs_cluster_offload_strategy[rows_index][ue_bs_min_index] = 1
            Ttr_ue_bs_cluster[rows_index][ue_bs_min_index] = ue_cluster[rows_index].data_size / ue_bs_clusters_communicate_rate[rows_index][ue_bs_min_index]
            bs_cluster[ue_bs_min_index].cpu_task += ue_cluster[rows_index].cpu_task
            bs_cluster[ue_bs_min_index].data_size += ue_cluster[rows_index].data_size
            ue_cluster[rows_index].cpu_task -= ue_cluster[rows_index].cpu_task
            ue_cluster[rows_index].data_size -= ue_cluster[rows_index].data_size
    ue_uav_and_ue_bs_cluster_offload_strategy = np.hstack((ue_uav_cluster_offload_strategy, ue_bs_cluster_offload_strategy))
    Ttr_ue_uav_and_ue_bs_cluster = np.hstack((Ttr_ue_uav_cluster, Ttr_ue_bs_cluster))
    return ue_uav_and_ue_bs_cluster_offload_strategy, Ttr_ue_uav_and_ue_bs_cluster


# 此处为无人机集群上的任务是否进行卸载的卸载策略，当前假定卸载策略仅与任务量大小有关；同时进行了各无人机 data_size、cpu_task 的卸载变化，以及卸载传输时延 Ttr 的计算
# 版本迭代后 uav 卸载策略直接在 get_strategy_and_Ttr_by_action_for_maddpg() 方法中计算，已弃用
def get_uav_bs_cluster_offload_strategy_and_Ttr(uav_cluster, bs_cluster):
    uav_bs_cluster_offload_strategy = np.zeros((len(uav_cluster), len(bs_cluster)))
    Ttr_uav_bs_cluster = np.zeros((len(uav_cluster), len(bs_cluster)))
    uav_bs_cluster_distance = get_clusters_distance(uav_cluster, bs_cluster)
    uav_bs_clusters_communicate_rate = get_uav_bs_communicate_rate(uav_cluster, bs_cluster)
    rows_index = -1
    for rows in uav_bs_cluster_distance:
        rows_index += 1
        if uav_cluster[rows_index].cpu_task >= 1000:
            uav_bs_min_index = (np.argwhere(rows == min(rows)))[0][0]
            uav_bs_cluster_offload_strategy[rows_index][uav_bs_min_index] = 1
            Ttr_uav_bs_cluster[rows_index][uav_bs_min_index] = uav_cluster[rows_index].data_size / uav_bs_clusters_communicate_rate[rows_index][uav_bs_min_index]
            bs_cluster[uav_bs_min_index].cpu_task += uav_cluster[rows_index].cpu_task
            bs_cluster[uav_bs_min_index].data_size += uav_cluster[rows_index].data_size
            uav_cluster[rows_index].cpu_task -= uav_cluster[rows_index].cpu_task
            uav_cluster[rows_index].data_size -= uav_cluster[rows_index].data_size
    return uav_bs_cluster_offload_strategy, Ttr_uav_bs_cluster


# 无人机飞行能耗，单位 焦耳/米 （在速度为 velocity 的前提下）
# 根据 ADVISOR-004 论文中的公式（1），以及参考文献004-[23]中的公式（8）（61）+ 附录TABLE I，进行建模计算
def get_power_cost_per_meter(velocity):
    return 580.65 * (1 + 0.000075 * velocity ** 2) + \
           790.67 * ((1 + velocity ** 4 / (103.68 ** 2)) ** (1/2) - velocity ** 2 / 103.68) ** (1/2) + \
           0.00726 * velocity ** 3


# 决定 ue 或 uav 上是否有任务要在其本地执行的策略
# 版本迭代后 ue 或 uav 上是否有任务要在其本地执行的策略直接在 get_strategy_and_Ttr_by_action_for_maddpg() 方法中计算，已弃用
def get_ue_or_uav_local_strategy(cluster_offload_strategy):
    local_strategy = []
    for rows in cluster_offload_strategy:
        flag = 0
        ue_xor = 0
        for cols in rows:
            flag += 1
            ue_xor += cols
        if flag >= len(rows) and ue_xor == 0:
            local_strategy.append(1)
        else:
            local_strategy.append(0)
    return local_strategy


# 决定 bs 上是否有任务要在其本地执行的策略
# 版本迭代后 bs 上是否有任务要在其本地执行的策略直接在 get_strategy_and_Ttr_by_action_for_maddpg() 方法中计算，已弃用
def get_bs_local_strategy(bs_cluster):
    local_strategy = []
    for items in bs_cluster:
        if items.cpu_task > 0:
            local_strategy.append(1)
        else:
            local_strategy.append(0)
    return local_strategy


# 根据强化学习输出的动作 action 值确定卸载策略；同时进行了各智能体 data_size、cpu_task 的卸载变化，以及卸载传输时延 Ttr 的计算

# 当前该方法针对 nue、nuav、nbs 的场景
def get_strategy_and_Ttr_by_action_for_maddpg(ue_cluster, uav_cluster, bs_cluster, building_cluster, action):
    ue_uav_cluster_offload_strategy = np.zeros((len(ue_cluster), len(uav_cluster)))
    ue_bs_cluster_offload_strategy = np.zeros((len(ue_cluster), len(bs_cluster)))
    uav_bs_cluster_offload_strategy = np.zeros((len(uav_cluster), len(bs_cluster)))
    ue_local_strategy = np.zeros(len(ue_cluster))
    uav_local_strategy = np.zeros(len(uav_cluster))
    bs_local_strategy = np.zeros(len(bs_cluster))
    Ttr_ue_uav_cluster = np.zeros((len(ue_cluster), len(uav_cluster)))
    Ttr_ue_bs_cluster = np.zeros((len(ue_cluster), len(bs_cluster)))
    Ttr_uav_bs_cluster = np.zeros((len(uav_cluster), len(bs_cluster)))
    ue_uav_clusters_communicate_rate = get_ue_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster)
    ue_bs_clusters_communicate_rate = get_ue_bs_communicate_rate(ue_cluster, bs_cluster)
    uav_bs_clusters_communicate_rate = get_uav_bs_communicate_rate(uav_cluster, bs_cluster)

    division_num_of_one_ue = (1 + len(uav_cluster)+len(bs_cluster) + len(uav_cluster) * len(bs_cluster))  # 关注 2ue、1uav、2bs 的场景，division_num_of_one_ue = 6
    division = 1 / division_num_of_one_ue
    # 可满足任意数量 ue、uav、bs 的场景
    for i in range(0, len(ue_cluster)):
        if action[i] <= 0 + division * 1:
            ue_local_strategy[i] = 1
        elif action[i] <= 0 + division * (1 + len(uav_cluster)):
            for j in range(0, len(uav_cluster)):
                if action[i] <= 0 + division * (1 + j + 1):
                    ue_uav_cluster_offload_strategy[i][j] = 1
                    Ttr_ue_uav_cluster[i][j] = ue_cluster[i].data_size / ue_uav_clusters_communicate_rate[i][j]
                    uav_cluster[j].cpu_task += ue_cluster[i].cpu_task
                    uav_cluster[j].data_size += ue_cluster[i].data_size
                    ue_cluster[i].cpu_task -= ue_cluster[i].cpu_task
                    ue_cluster[i].data_size -= ue_cluster[i].data_size
                    uav_local_strategy[j] = 1
                    break
        elif action[i] <= 0 + division * (1 + len(uav_cluster) + len(bs_cluster)):
            for j in range(0, len(bs_cluster)):
                if action[i] <= 0 + division * (1 + len(uav_cluster) + j + 1):
                    ue_bs_cluster_offload_strategy[i][j] = 1
                    Ttr_ue_bs_cluster[i][j] = ue_cluster[i].data_size / ue_bs_clusters_communicate_rate[i][j]
                    bs_cluster[j].cpu_task += ue_cluster[i].cpu_task
                    bs_cluster[j].data_size += ue_cluster[i].data_size
                    ue_cluster[i].cpu_task -= ue_cluster[i].cpu_task
                    ue_cluster[i].data_size -= ue_cluster[i].data_size
                    bs_local_strategy[j] = 1
                    break
        elif action[i] <= 0 + division * (1 + len(uav_cluster) + len(bs_cluster) + len(uav_cluster) * len(bs_cluster)):
            try:
                for j in range(0, len(uav_cluster)):
                    for k in range(0, len(bs_cluster)):
                        if action[i] <= 0 + division * (1 + len(uav_cluster) + len(bs_cluster) + j * 2 + k + 1):
                            ue_uav_cluster_offload_strategy[i][j] = 1
                            Ttr_ue_uav_cluster[i][j] = ue_cluster[i].data_size / ue_uav_clusters_communicate_rate[i][j]
                            uav_cluster[j].cpu_task += ue_cluster[i].cpu_task
                            uav_cluster[j].data_size += ue_cluster[i].data_size
                            ue_cluster[i].cpu_task -= ue_cluster[i].cpu_task
                            ue_cluster[i].data_size -= ue_cluster[i].data_size
                            uav_bs_cluster_offload_strategy[j][k] = 1
                            Ttr_uav_bs_cluster[j][k] = uav_cluster[j].data_size / uav_bs_clusters_communicate_rate[j][k]
                            bs_cluster[k].cpu_task += uav_cluster[j].cpu_task
                            bs_cluster[k].data_size += uav_cluster[j].data_size
                            uav_cluster[j].cpu_task -= uav_cluster[j].cpu_task
                            uav_cluster[j].data_size -= uav_cluster[j].data_size
                            bs_local_strategy[k] = 1
                            raise StopIteration
            except StopIteration:
                pass

    ue_uav_and_ue_bs_cluster_offload_strategy = np.hstack((ue_uav_cluster_offload_strategy, ue_bs_cluster_offload_strategy))
    Ttr_ue_uav_and_ue_bs_cluster = np.hstack((Ttr_ue_uav_cluster, Ttr_ue_bs_cluster))

    return ue_uav_and_ue_bs_cluster_offload_strategy, Ttr_ue_uav_and_ue_bs_cluster, \
           uav_bs_cluster_offload_strategy, Ttr_uav_bs_cluster, \
           ue_local_strategy, uav_local_strategy, bs_local_strategy


# 集群优化目标值的微分；数据 回传的时延暂时不考虑
def get_cluster_target_slice_for_maddpg(ue_cluster, uav_cluster, bs_cluster, building_cluster, action):

    ue_uav_and_ue_bs_cluster_offload_strategy, Ttr_ue_uav_and_ue_bs_cluster, \
    uav_bs_cluster_offload_strategy, Ttr_uav_bs_cluster, \
    ue_local_strategy, uav_local_strategy, bs_local_strategy = \
        get_strategy_and_Ttr_by_action_for_maddpg(ue_cluster, uav_cluster, bs_cluster, building_cluster, action)

    Tc_ue_cluster = get_Tc_agent_cluster(ue_cluster)
    Tc_uav_cluster = get_Tc_agent_cluster(uav_cluster)
    Tc_bs_cluster = get_Tc_agent_cluster(bs_cluster)

    return np.trace(np.dot(ue_uav_and_ue_bs_cluster_offload_strategy, np.transpose(Ttr_ue_uav_and_ue_bs_cluster))) + \
           np.trace(np.dot(uav_bs_cluster_offload_strategy, np.transpose(Ttr_uav_bs_cluster))) + \
           np.dot(ue_local_strategy, np.transpose(Tc_ue_cluster)) + \
           np.dot(uav_local_strategy, np.transpose(Tc_uav_cluster)) + \
           np.dot(bs_local_strategy, np.transpose(Tc_bs_cluster))


def get_strategy_by_action_for_maddpg(ue_cluster, uav_cluster, bs_cluster, building_cluster, irs_cluster, action, ue_index):
    #  action 为 0.25 或者 0.75
    # 基站处理的任务量
    bs_cluster_task = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机与基站的卸载策略
    uav_bs_cluster_offload_strategy = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机总能耗
    E_uav = 0
    ue_uav_cluster_offload_strategy = np.zeros((len(uav_cluster)))
    #  ue_uav端的总时延
    T_ue_de = 0
    T_ue_uav_cluster = np.zeros(len(uav_cluster))
    ue_uav_clusters_communicate_rate = get_gu_uav_communicate_rate(ue_cluster, uav_cluster, building_cluster,
                                                                   irs_cluster)
    division_num_of_one_ue = (1 + len(uav_cluster))  # 维度为 无人机个数 + 用户
    division = 1 / division_num_of_one_ue            # 0.5
    # 每个用户根据action判断卸载策略
    if action <= 0 + division * 1:
        ue_local_strategy = 1  # UE本地处理置1
        T_ue_uav_cluster += 1e9 / ue_cluster[ue_index].cpu_power
    elif action <= 0 + division * (1 + len(uav_cluster)):
        # 没考虑选择哪个无人机，场景目前只有一个无人机
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

def get_uav_bs_cluster_offload_strategy_and_T(uav_cluster, bs_cluster, cpu_task, data_size):
    # 基站处理的任务量
    bs_cluster_task_ = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机与基站的卸载策略
    uav_bs_cluster_offload_strategy = np.zeros((len(uav_cluster), len(bs_cluster)))
    # 无人机总能耗
    E_uav_ = 0
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
        E_uav_de += chi * (cpu_task) ** 2 * uav_cluster[rows_index].cpu_power * 10
        # 传输能耗
        E_uav_tr += Power_uav * data_size / (np.array(uav_bs_clusters_communicate_rate))
        if T_uav_de < np.min(T_uav_tr_bs_cluster):
            T_uav_bs_cluster = T_uav_de
            E_uav_ += E_uav_de
        else:
            T_uav_bs_cluster = np.min(T_uav_tr_bs_cluster)
            min_delay_bs_index = np.argmin(T_uav_tr_bs_cluster, axis=None)
            min_list = np.unravel_index(min_delay_bs_index, T_uav_tr_bs_cluster.shape, order='C')
            # 最小值的位置为
            min_location = min_list[1]
            uav_bs_cluster_offload_strategy[0][min_location] = 1.0
            bs_cluster_task_[0][min_location] += cpu_task
            E_uav_ += E_uav_tr[0][min_location]
    return uav_bs_cluster_offload_strategy, T_uav_bs_cluster, bs_cluster_task_, E_uav_

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