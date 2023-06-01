from agents import Agent, UE, BS, UAV, Building, IRS
from ma_ddpg.multiagent_envs.env_functions import *
import random
import numpy as np


class World(object):
    M = 100
    phase_shift = 100
    B = 8000  # 8MHz
    Xi = mt.pow(10, (3 / 10))
    white_noise = 10
    Power_ue = 3 * mt.pow(10, 3)  # 300mW
    Power_uav = 5 * mt.pow(10, 3)  # 500mW
    N_0 = mt.pow(10, (-169 / 10)) * (0.1 * mt.pow(10, 3)) / B
    chi = 1e-26

    ue1 = UE.UE(140, 165, 0)
    ue2 = UE.UE(170, 170, 0)
    ue3 = UE.UE(200, 150, 0)
    ue4 = UE.UE(380, 350, 0)
    ue5 = UE.UE(520, 200, 0)

    uav1 = UAV.UAV(0, 200, 1, 1500000)  # 根据参考论文，无人机初始能量为 500kJ，此处定为 1500kJ 是为了让仿真时无人机可以飞得久一点，到 90 多秒再降落
    irs1 = IRS.IRS(180, 180, 25)

    bs1 = BS.BS_local(160, 175, 20)
    bs2 = BS.BS(340, 320, 20)
    bs3 = BS.BS(400, 340, 20)
    bs4 = BS.BS(540, 210, 20)
    bs5 = BS.BS(580, 160, 20)

    building1 = Building.Building(100, 150, 0, 26, 14, 20)
    building2 = Building.Building(100, 180, 0, 26, 14, 20)
    building3 = Building.Building(130, 140, 0, 26, 14, 20)
    building4 = Building.Building(130, 180, 0, 26, 14, 20)
    building5 = Building.Building(160, 140, 0, 26, 14, 20)
    building6 = Building.Building(180, 180, 0, 26, 14, 25)
    building7 = Building.Building(300, 300, 0, 26, 14, 20)
    building8 = Building.Building(300, 330, 0, 26, 14, 20)
    building9 = Building.Building(300, 360, 0, 26, 14, 20)
    building10 = Building.Building(360, 300, 0, 26, 14, 20)
    building11 = Building.Building(360, 330, 0, 26, 14, 20)
    building12 = Building.Building(360, 360, 0, 26, 14, 20)
    building13 = Building.Building(520, 180, 0, 26, 14, 20)
    building14 = Building.Building(560, 200, 0, 26, 14, 20)

    # 其他住房建筑1
    building15 = Building.Building(300, 150, 0, 8, 6, 40)
    building16 = Building.Building(320, 150, 0, 8, 6, 40)
    building17 = Building.Building(340, 150, 0, 8, 6, 40)
    building18 = Building.Building(360, 150, 0, 8, 6, 40)
    building19 = Building.Building(380, 150, 0, 8, 6, 40)
    building20 = Building.Building(400, 150, 0, 8, 6, 40)

    building21 = Building.Building(300, 170, 0, 8, 6, 40)
    building22 = Building.Building(320, 170, 0, 8, 6, 40)
    building23 = Building.Building(340, 170, 0, 8, 6, 40)
    building24 = Building.Building(360, 170, 0, 8, 6, 40)
    building25 = Building.Building(380, 170, 0, 8, 6, 40)
    building26 = Building.Building(400, 170, 0, 8, 6, 40)

    building27 = Building.Building(300, 190, 0, 8, 6, 40)
    building28 = Building.Building(320, 190, 0, 8, 6, 40)
    building29 = Building.Building(340, 190, 0, 8, 6, 40)
    building30 = Building.Building(360, 190, 0, 8, 6, 40)
    building31 = Building.Building(380, 190, 0, 8, 6, 40)
    building32 = Building.Building(400, 190, 0, 8, 6, 40)

    # 其他住房建筑2
    building33 = Building.Building(300, 210, 0, 8, 6, 40)
    building34 = Building.Building(320, 210, 0, 8, 6, 40)
    building35 = Building.Building(340, 210, 0, 8, 6, 40)
    building36 = Building.Building(360, 210, 0, 8, 6, 40)
    building37 = Building.Building(380, 210, 0, 8, 6, 40)
    building38 = Building.Building(400, 210, 0, 8, 6, 40)

    building39 = Building.Building(300, 230, 0, 8, 6, 40)
    building40 = Building.Building(320, 230, 0, 8, 6, 40)
    building41 = Building.Building(340, 230, 0, 8, 6, 40)
    building42 = Building.Building(360, 230, 0, 8, 6, 40)
    building43 = Building.Building(380, 230, 0, 8, 6, 40)
    building44 = Building.Building(400, 230, 0, 8, 6, 40)

    building45 = Building.Building(300, 250, 0, 8, 6, 40)
    building46 = Building.Building(320, 250, 0, 8, 6, 40)
    building47 = Building.Building(340, 250, 0, 8, 6, 40)
    building48 = Building.Building(360, 250, 0, 8, 6, 40)
    building49 = Building.Building(380, 250, 0, 8, 6, 40)
    building50 = Building.Building(400, 250, 0, 8, 6, 40)

    # 其他住房建筑3
    building51 = Building.Building(300, 130, 0, 8, 6, 40)
    building52 = Building.Building(320, 130, 0, 8, 6, 40)
    building53 = Building.Building(340, 130, 0, 8, 6, 40)
    building54 = Building.Building(360, 130, 0, 8, 6, 40)
    building55 = Building.Building(380, 130, 0, 8, 6, 40)
    building56 = Building.Building(400, 130, 0, 8, 6, 40)

    building57 = Building.Building(300, 110, 0, 8, 6, 40)
    building58 = Building.Building(320, 110, 0, 8, 6, 40)
    building59 = Building.Building(340, 110, 0, 8, 6, 40)
    building60 = Building.Building(360, 110, 0, 8, 6, 40)
    building61 = Building.Building(380, 110, 0, 8, 6, 40)
    building62 = Building.Building(400, 110, 0, 8, 6, 40)

    building63 = Building.Building(300, 90, 0, 8, 6, 40)
    building64 = Building.Building(320, 90, 0, 8, 6, 40)
    building65 = Building.Building(340, 90, 0, 8, 6, 40)
    building66 = Building.Building(360, 90, 0, 8, 6, 40)
    building67 = Building.Building(380, 90, 0, 8, 6, 40)
    building68 = Building.Building(400, 90, 0, 8, 6, 40)

    # 其他住房建筑4
    building69 = Building.Building(100, 310, 0, 8, 6, 40)
    building70 = Building.Building(120, 310, 0, 8, 6, 40)
    building71 = Building.Building(140, 310, 0, 8, 6, 40)
    building72 = Building.Building(160, 310, 0, 8, 6, 40)
    building73 = Building.Building(180, 310, 0, 8, 6, 40)
    building74 = Building.Building(200, 310, 0, 8, 6, 40)

    building75 = Building.Building(100, 330, 0, 8, 6, 40)
    building76 = Building.Building(120, 330, 0, 8, 6, 40)
    building77 = Building.Building(140, 330, 0, 8, 6, 40)
    building78 = Building.Building(160, 330, 0, 8, 6, 40)
    building79 = Building.Building(180, 330, 0, 8, 6, 40)
    building80 = Building.Building(200, 330, 0, 8, 6, 40)

    # 其他住房建筑5
    building81 = Building.Building(100, 350, 0, 8, 6, 40)
    building82 = Building.Building(120, 350, 0, 8, 6, 40)
    building83 = Building.Building(140, 350, 0, 8, 6, 40)
    building84 = Building.Building(160, 350, 0, 8, 6, 40)
    building85 = Building.Building(180, 350, 0, 8, 6, 40)
    building86 = Building.Building(200, 350, 0, 8, 6, 40)

    building87 = Building.Building(100, 370, 0, 8, 6, 40)
    building88 = Building.Building(120, 370, 0, 8, 6, 40)
    building89 = Building.Building(140, 370, 0, 8, 6, 40)
    building90 = Building.Building(160, 370, 0, 8, 6, 40)
    building91 = Building.Building(180, 370, 0, 8, 6, 40)
    building92 = Building.Building(200, 370, 0, 8, 6, 40)

    # 其他住房建筑6
    building93 = Building.Building(100, 390, 0, 8, 6, 40)
    building94 = Building.Building(120, 390, 0, 8, 6, 40)
    building95 = Building.Building(140, 390, 0, 8, 6, 40)
    building96 = Building.Building(160, 390, 0, 8, 6, 40)
    building97 = Building.Building(180, 390, 0, 8, 6, 40)
    building98 = Building.Building(200, 390, 0, 8, 6, 40)

    building99 = Building.Building(100, 210, 0, 8, 6, 40)
    building100 = Building.Building(120, 210, 0, 8, 6, 40)
    building101 = Building.Building(140, 210, 0, 8, 6, 40)
    building102 = Building.Building(160, 210, 0, 8, 6, 40)
    building103 = Building.Building(180, 210, 0, 8, 6, 40)
    building104 = Building.Building(200, 210, 0, 8, 6, 40)

    building105 = Building.Building(100, 230, 0, 8, 6, 40)
    building106 = Building.Building(120, 230, 0, 8, 6, 40)
    building107 = Building.Building(140, 230, 0, 8, 6, 40)
    building108 = Building.Building(160, 230, 0, 8, 6, 40)
    building109 = Building.Building(180, 230, 0, 8, 6, 40)
    building110 = Building.Building(200, 230, 0, 8, 6, 40)

    building111 = Building.Building(100, 250, 0, 8, 6, 40)
    building112 = Building.Building(120, 250, 0, 8, 6, 40)
    building113 = Building.Building(140, 250, 0, 8, 6, 40)
    building114 = Building.Building(160, 250, 0, 8, 6, 40)
    building115 = Building.Building(180, 250, 0, 8, 6, 40)
    building116 = Building.Building(200, 250, 0, 8, 6, 40)

    irs_cluster = [irs1]
    ue_cluster = [ue1, ue2, ue3, ue4, ue5]
    uav_cluster = [uav1]
    bs_cluster = [bs1, bs2, bs3, bs4, bs5]
    building_cluster = [building1, building2, building3, building4, building5, building6, building7, building8,
                        building9, building10, building11, building12, building13, building14, building15, building16,
                        building17, building18,
                        building19, building20, building21, building22, building23, building24, building25, building26,
                        building27, building28,
                        building29, building30, building31, building32, building33, building34, building35, building36,
                        building37, building38,
                        building39, building40, building41, building42, building43, building44, building45, building46,
                        building47, building48,
                        building49, building50, building51, building52, building53, building54, building55, building56,
                        building57, building58,
                        building59, building60,
                        building61, building62, building63, building64, building65, building66, building67, building68,
                        building69, building70,
                        building71, building72, building73, building74, building75, building76, building77, building78,
                        building79, building80,
                        building81, building82, building83, building84, building85, building86, building87, building88,
                        building89, building90,
                        building91, building92, building93, building94, building95, building96, building97, building98,
                        building99, building100,
                        building101, building102, building103, building104, building105, building106, building107,
                        building108, building109, building110,
                        building111, building112, building113, building114, building115, building116]

    omega1 = 0.05
    omega2 = 1
    # 任务生成方式(固定)
    for ue in ue_cluster:
        ue.data_size = 1e6
        ue.cpu_task = 1e9

    uav_cluster_x = []
    uav_cluster_y = []
    uav_cluster_z = []
    for uav in uav_cluster:
        uav_cluster_x.append([uav.x])
        uav_cluster_y.append([uav.y])
        uav_cluster_z.append([uav.z])
    ue_num = len(ue_cluster)
    uav_num = len(uav_cluster)
    bs_num = len(bs_cluster)
    action_dim_for_uav = 3  # uav 的 action 定 3 个，依次为：uav飞行仰角，uav飞行水平夹角，uav飞行速度
    #
    situation_n = ["" for m in range(len(uav_cluster))]

    is_terminal_n = [False for k in range(len(uav_cluster))]
    uav_pop_count = 0

    # # modified 2 此处在仅将 ue 作为智能体时需要保留，其余场景需要注释掉
    # is_terminal_for_only_ue = [True for j in range(len(ue_cluster))]
    # step_count_for_only_ue = 0

    def __init__(self):
        self.start_state = []
        self.state = self.start_state

    def reset_env(self):
        # ue 初始化
        for ue in self.ue_cluster:
            ue.data_size = 1e6
            ue.cpu_task = 1e9
        # uav 初始化
        self.uav1 = UAV.UAV(0, 100, 1, 1500000)
        self.uav_cluster = [self.uav1]

        # bs 初始化
        self.bs1 = BS.BS_local(160, 175, 20)
        self.bs2 = BS.BS(340, 320, 20)
        self.bs3 = BS.BS(400, 340, 20)
        self.bs4 = BS.BS(440, 210, 20)
        self.bs5 = BS.BS(480, 160, 20)
        self.bs_cluster = [self.bs1, self.bs2, self.bs3, self.bs4, self.bs5]
        # reset 时，要重新绘制无人机飞行轨迹，因此需要重置记录无人机飞行轨迹的数组
        self.uav_cluster_x = []
        self.uav_cluster_y = []
        self.uav_cluster_z = []
        # 绘制起飞段图形
        # 按照当前实现的代码逻辑，如果一开始无人机处于一个较低的高度如 1m ，则进行路径规划时很容易就撞到地面，预计要经过很长时间的训练才可以完成起飞路线的规划
        # 因此在进行训练的时候，一开始就通过调用 launching() 方法使无人机处于一个较高高度如 90m ，在该情况下进行的路径规划不会太过于容易就撞到地面
        for uav in self.uav_cluster:
            self.uav_cluster_x.append([uav.x])
            self.uav_cluster_y.append([uav.y])
            self.uav_cluster_z.append([uav.z])
        # 起飞
        self.uav_cluster = launching(self)
        self.uav1 = self.uav_cluster[0]
        self.situation_n = ["" for m in range(len(self.uav_cluster))]
        self.is_terminal_n = [False for k in range(len(self.uav_cluster))]
        self.uav_pop_count = 0
        # # modified 2 此处在仅将 ue 作为智能体时需要保留，其余场景需要注释掉
        # self.is_terminal_for_only_ue = [True for k in range(len(self.ue_cluster))]
        # self.step_count_for_only_ue = 0

    def _get_obs(self, agent):
        self.state = []
        if isinstance(agent, UAV.UAV):
            #  arr，values都将先展平成一维数组,然后沿着axis=0的方向在arr后添加values
            self.state = np.append(self.state, agent.xyz)
            self.state = np.append(self.state, agent.energy)
        elif isinstance(agent, UE.UE):
            self.state = np.append(self.state, agent.data_size)
            self.state = np.append(self.state, agent.cpu_task)
        return self.state

    def reset(self):
        self.reset_env()
        self.state = []

    # 重置 ue、uav、bs 任务
    def reset2(self):
        for ue in self.ue_cluster:
            ue.data_size = 1e9
            ue.cpu_task = 1e6
        # reset 时，将 uav、bs 的 data_size、cpu_task 初始化为0，因为在卸载过程中 uav、bs 上也可能有了一定的任务量
        for uav in self.uav_cluster:
            uav.data_size = 0
            uav.cpu_task = 0
        for bs in self.bs_cluster:
            bs.data_size = 0
            bs.cpu_task = 0

    def step(self, agents, action_n):
        bs_task_sum = np.zeros((len(self.uav_cluster), len(self.bs_cluster)))
        reward_n = []
        delay_n = []
        situation_n = ["" for m in range(len(self.uav_cluster))]
        is_terminal_n = [False for k in range(len(self.uav_cluster))]
        for agent_index in range(0, 2):
            uav_index = -1
            ue_index = -1
            for agent in agents[agent_index]:
                if isinstance(agent, UAV.UAV):
                    uav_index += 1
                    reward, is_terminal_n, situation_n = self.step_for_uav(action_n, uav_index)
                    reward_n.append(reward)
                    if all(is_terminal_n):
                        # modified 2 此处在仅将 uav 作为智能体时需要注释掉，其余场景需要保留
                        delay_n = [0 for k in range(len(self.ue_cluster))]
                        for i in range(len(self.ue_cluster)):
                            reward_n.append(0)
                        return reward_n, is_terminal_n, delay_n, situation_n
                elif isinstance(agent, UE.UE):
                    ue_index += 1
                    reward, delay, energy, bs_task_sum = self.step_for_ue(action_n, uav_index, ue_index, bs_task_sum)
                    reward_n.append(reward)
                    delay_n.append(delay)
                    self.uav_cluster[0].energy -= energy

        self.reset2()  # 重置 ue、uav、bs 任务

        return reward_n, is_terminal_n, delay_n, situation_n

    def step_for_ue(self, action_n, uav_index, ue_index, bs_task_sum):
        numerator = 0
        # 将 one-hot 形式的 ue 动作 action 转变为单一数值
        agent_index = uav_index + ue_index + 2
        array_temp = action_n[agent_index].tolist()
        max_value = max(array_temp)
        max_index = array_temp.index(max_value)
        # 因为这个值可能落在前面一个数，所以加上一小段让他准确
        action_value = max_index / len(action_n[agent_index]) + 0.5 / len(action_n[agent_index])
        # if agent_index < len(self.ue_cluster):
        #     agent_index += 1
        # 记录时延
        uav_bs_cluster_offload_strategy, delay_, bs_task, energy = get_strategy_by_action_for_maddpg(self.ue_cluster,
                                                                                                     self.uav_cluster,
                                                                                                     self.bs_cluster,
                                                                                                     self.building_cluster,
                                                                                                     self.irs_cluster,
                                                                                                     action_value,
                                                                                                     ue_index)
        bs_task_sum += bs_task
        # 计算每部分的比值
        if action_value == 0.75:
            ratio_task = np.zeros((1, len(self.bs_cluster)))
            for i in range(len(self.bs_cluster)):
                ratio_task[0][i] += bs_task_sum[0][i] / np.sum(bs_task_sum)
            # f的分子
            for i in range(len(ratio_task[0])):
                numerator += math.pow(ratio_task[0][i], 2)
            numerator *= len(ratio_task[0])
            f = 1 / numerator
        reward_ = - delay_ * 1000
        delay = delay_[0]
        reward = np.mean(reward_)
        # reward = (f / omega1 * energy + omega2 * delay) * 100
        return reward, delay, energy, bs_task_sum

    def step_for_uav(self, action_n, agent_index):
        reward_n = [0 for m in range(len(self.uav_cluster))]
        presupposition_point = Agent.Agent(600, 400, 30)  # 预设点
        uav_index = -1
        for uav in self.uav_cluster:
            uav_index += 1
            if uav_index == agent_index:
                theta, fai, velocity = 0, 0, 0
                if uav_index == 0:
                    theta = action_n[uav_index][0] * math.pi * 2 + random.uniform(0.45, 0.55) * math.pi  # 仰角
                    fai = action_n[uav_index][1] * math.pi * 2 + random.uniform(0.1, 0.2) * math.pi  # 水平夹角
                    velocity = (action_n[uav_index][2] + 0.000001) * 10  # 飞行速度
                # 为帮助算法合理收敛，设置理想降落点在预设点正下方
                uav0 = UAV.UAV(presupposition_point.x, presupposition_point.y, 1, 1500000)
                uav0_z = UAV.UAV(presupposition_point.x, presupposition_point.y, uav.z, 1500000)
                uav0_xy = UAV.UAV(uav.x, uav.y, 1, 1500000)
                uav_uav0_distance_temp = get_uav_uav0_distance(uav, uav0)
                uav_z_temp = uav.z
                uav_presupposition_point_distance_temp = get_clusters_distance([uav], [presupposition_point])
                if uav.z > 1:
                    # 更新下一时刻无人机的状态
                    uav.flight_trace(theta, fai, velocity, 1)
                    uav.energy = uav.energy - get_power_cost_per_meter(abs(velocity)) * abs(velocity) * 1
                # 无人机与终点的距离
                uav_uav0_distance = get_uav_uav0_distance(uav, presupposition_point)
                if uav_uav0_distance <= 700:
                    reward_n[uav_index] = reward_n[uav_index] + (700 - uav_uav0_distance) * 1
                    if uav_uav0_distance <= 650:
                        reward_n[uav_index] = reward_n[uav_index] + (700 - uav_uav0_distance) * 5
                        if uav_uav0_distance <= 600:
                            reward_n[uav_index] = reward_n[uav_index] + (700 - uav_uav0_distance) * 10
                            if uav_uav0_distance <= 550:
                                reward_n[uav_index] = reward_n[uav_index] + (700 - uav_uav0_distance) * 50
                                if uav_uav0_distance <= 500:
                                    reward_n[uav_index] = reward_n[uav_index] + (700 - uav_uav0_distance) * 100
                                    if uav_uav0_distance <= 450:
                                        reward_n[uav_index] = reward_n[uav_index] + (
                                                450 - uav_uav0_distance) * 500
                                        if uav_uav0_distance <= 400:
                                            reward_n[uav_index] = reward_n[uav_index] + (400 - uav_uav0_distance) * 1000
                                            if uav_uav0_distance <= 350:
                                                reward_n[uav_index] = reward_n[uav_index] + (
                                                        350 - uav_uav0_distance) * 5000
                                                if uav_uav0_distance <= 300:
                                                    reward_n[uav_index] = reward_n[uav_index] + (
                                                            300 - uav_uav0_distance) * 10000
                                                    if uav_uav0_distance <= 250:
                                                        reward_n[uav_index] = reward_n[uav_index] + (
                                                                250 - uav_uav0_distance) * 15000
                                                        if uav_uav0_distance <= 150:
                                                            reward_n[uav_index] = reward_n[uav_index] + (
                                                                    150 - uav_uav0_distance) * 20000
                                                            if uav_uav0_distance <= 100:
                                                                reward_n[uav_index] = reward_n[uav_index] + (
                                                                            100 - uav_uav0_distance) * 25000
                                                                if uav_uav0_distance <= 50:
                                                                    reward_n[uav_index] = reward_n[uav_index] + (
                                                                            50 - uav_uav0_distance) * 25000
                else:
                    reward_n[uav_index] = reward_n[uav_index] - 10000  # 给 reward 额外减去一个偏置值
                    # 终止当前 step & episode
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "go_away"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n
                # 无人机达到降落范围
                if 575 <= uav.x <= 625 and 375 <= uav.y <= 425:
                    reward_n[uav_index] = reward_n[uav_index] + 100000  # 达到降落范围
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "well landing"
                    landing(self, uav, uav_index)
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n
                # 无人机飞行飞出边境
                if uav.x >= 650 or uav.x < -5 or uav.y >= 450 or uav.y < 0:
                    reward_n[uav_index] = reward_n[uav_index] - 10000  # 给 reward 额外减去一个超越边境的惩罚
                    # 终止当前 step & episode
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "beyond the border"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n
                # 坠机（无人机与建筑物相撞或无人机之间相撞）
                if uav_crash_judgement(self.uav_cluster, self.building_cluster):
                    reward_n[uav_index] = reward_n[uav_index] - 10000  # 给 reward 额外减去一个偏置值
                    # 终止当前 step & episode
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "crash"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n
                # 坠机（越过 1m 降落高度标准线时速度不为 0，可能撞向地面）
                if uav.z < 1:
                    reward_n[uav_index] = reward_n[uav_index] - 10000  # 给 reward 额外减去一个偏置值
                    # print("down")
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "down"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n
                # 坠机（未降落时能量即将耗尽，已不足以支撑降落）
                if uav.z >= 1 and uav.energy <= get_power_cost_per_meter(3) * (uav.z - 1):
                    reward_n[uav_index] = reward_n[uav_index] - 10000  # 给 reward 额外减去一个偏置值
                    # 终止当前 step & episode
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "no_energy"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n
                # 若进行原地降落不会与建筑物相撞，可以进入降落程序
                if uav.z >= 1 and (not uav_crash_judgement([uav0_xy], self.building_cluster)):
                    # 已经降落，且上一个step()中 uav 高度大于 1，说明是未经过降落程序自动达到降落高度
                    if uav.z == 1 and uav_z_temp > 1:
                        if uav.energy <= 1500000 * 0.3:  # 剩余能量较少
                            # 如果此时处于降落区域内给予额外奖励，否则给予额外惩罚
                            if get_uav_uav0_distance(uav, uav0_z) <= 50:
                                reward_n[uav_index] = reward_n[uav_index] + 50000  # 给 reward 额外加上一个偏置值
                                self.situation_n[uav_index] = "well_landing"
                            else:
                                reward_n[uav_index] = reward_n[uav_index]  # 给 reward 额外减去一个偏置值
                                self.situation_n[uav_index] = "no energy landing"
                        else:  # 剩余能量充足
                            # 如果此时处于降落区域内给予较少的额外奖励，否则给予较多的额外惩罚
                            if get_uav_uav0_distance(uav, uav0_z) <= 50:
                                reward_n[uav_index] = reward_n[uav_index] + 30000  # 给 reward 额外加上一个偏置值
                            else:
                                reward_n[uav_index] = reward_n[uav_index]  # 给 reward 额外减去一个偏置值
                            self.situation_n[uav_index] = "landing"
                        self.uav_pop_count += 1  # 当前 uav 已着陆，uav_pop_count 计数 + 1
                        self.is_terminal_n[uav_index] = True
                        if self.uav_pop_count == len(self.uav_cluster):  # 全部 uav 都已着陆
                            # 终止当前 step & episode
                            self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                            return reward_n[uav_index], self.is_terminal_n, self.situation_n
                    # 已经降落，且上一个step()中 uav 高度等于 1，说明该 uav 是停在原地，不需要奖惩
                    if uav.z == 1 and uav_z_temp == 1:
                        self.uav_pop_count += 1  # 当前 uav 已着陆，uav_pop_count 计数 + 1
                        self.is_terminal_n[uav_index] = True
                        self.situation_n[uav_index] = "landing"
                        if self.uav_pop_count == len(self.uav_cluster):  # 全部 uav 都已着陆
                            # 终止当前 step & episode
                            self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                            return reward_n[uav_index], self.is_terminal_n, self.situation_n
                    # 暂未降落
                    if uav.z > 1:
                        if uav.energy <= 2 * get_power_cost_per_meter(3) * (
                                uav.z - 1):  # 能量即将耗尽(不足以 3m/s 原地降落所需能量的 2 倍)
                            # 如果此时处于降落区域内给予额外奖励，否则给予额外惩罚
                            if get_uav_uav0_distance(uav, uav0_z) <= 50:
                                reward_n[uav_index] = reward_n[uav_index] + 30000  # 给 reward 额外加上一个偏置值
                                self.situation_n[uav_index] = "well_landing"
                            else:
                                reward_n[uav_index] = reward_n[uav_index] - 1000 * get_uav_uav0_distance(uav,
                                                                                                         uav0)  # 给 reward 额外减去一个偏置值
                                self.situation_n[uav_index] = "landing"
                            landing(self, uav, uav_index)
                            self.uav_pop_count += 1  # 当前 uav 已着陆，uav_pop_count 计数 + 1
                            self.is_terminal_n[uav_index] = True
                            if self.uav_pop_count == len(self.uav_cluster):  # 全部 uav 都已着陆
                                # 终止当前 step & episode
                                self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                                return reward_n[uav_index], self.is_terminal_n, self.situation_n
                        else:  # 能量尚且充足，继续迭代寻找降落机会
                            pass
                else:  # 若进行原地降落会与建筑物相撞，继续迭代寻找降落机会
                    pass
                if uav.z > 150:
                    reward_n[uav_index] = reward_n[uav_index] - 10000  # 给 reward 额外减去一个偏置值
                    # 终止当前 step & episode
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "too_high"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n

                if uav.z < 50 and uav.z < uav_z_temp:  # uav 高度过低且继续向下飞
                    reward_n[uav_index] = reward_n[uav_index]  # reward 不变
                if uav.energy < 1500000 * 0.5 and get_uav_uav0_distance(uav,
                                                                        uav0) >= uav_uav0_distance_temp:  # uav 能量不足且继续远离降落点
                    reward_n[uav_index] = reward_n[uav_index] - 100 * (
                        get_uav_uav0_distance(uav, uav0))  # 给 reward 额外减去一个偏置值
                if uav.energy >= 1500000 * 0.5 and uav.z > 1 \
                        and get_clusters_distance([uav], [presupposition_point]) < \
                        uav_presupposition_point_distance_temp:  # uav 能量充足，在空中执行任务并靠近预设点位置
                    reward_n[uav_index] = reward_n[uav_index] + 100 * get_uav_uav0_distance(uav, uav0)
                self.uav_cluster_x[uav_index].append(uav.x)
                self.uav_cluster_y[uav_index].append(uav.y)
                self.uav_cluster_z[uav_index].append(uav.z)

        return reward_n[uav_index], self.is_terminal_n, self.situation_n
