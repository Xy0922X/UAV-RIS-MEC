from agents import Agent, UE, BS, UAV, Building
from ma_ddpg.multiagent_envs.env_functions import *
import numpy as np

class World(object):
    ue1 = UE.UE(8, 55, 0.5)
    ue2 = UE.UE(50, 25, 0.5)
    ue3 = UE.UE(15, 65, 0.5)
    ue4 = UE.UE(20, 45, 0.5)
    ue5 = UE.UE(65, 85, 0.5)
    uav1 = UAV.UAV(1, 1, 1, 1500000)  # 根据参考论文，无人机初始能量为 500kJ，此处定为 1500kJ 是为了让仿真时无人机可以飞得久一点，到 90 多秒再降落
    uav2 = UAV.UAV(90, 100, 1, 1500000)
    uav3 = UAV.UAV(90, 10, 1, 1500000)
    bs1 = BS.BS(10, 75, 50)
    bs2 = BS.BS(110, 85, 50)
    bs3 = BS.BS(110, 5, 50)
    bs4 = BS.BS(35, 5, 50)
    bs5 = BS.BS(60, 70, 50)
    building1 = Building.Building(10, 20, 0, 10, 10, 65)
    building2 = Building.Building(20, 20, 0, 10, 10, 50)
    building3 = Building.Building(55, 40, 0, 10, 10, 60)
    building4 = Building.Building(65, 40, 0, 10, 10, 30)
    building5 = Building.Building(55, 30, 0, 10, 10, 20)
    building6 = Building.Building(80, 70, 0, 10, 10, 30)
    building7 = Building.Building(80, 80, 0, 10, 10, 50)

    ue_cluster = [ue2]
    # ue_cluster = [ue1, ue2]
    # ue_cluster = [ue1, ue2, ue3, ue4, ue5]
    # uav_cluster = [uav1]
    # uav_cluster = [uav2]
    uav_cluster = [uav3]
    # uav_cluster = [uav1, uav2]
    # bs_cluster = [bs1]
    bs_cluster = [bs1, bs2]
    # bs_cluster = [bs1, bs2, bs3, bs4, bs5]
    # building_cluster = [building1, building2, building3, building4, building5, building6, building7]
    # building_cluster = [building1, building2]
    # building_cluster = [building7]
    building_cluster = []

    for ue in ue_cluster:
        ue.data_size = 1e7
        ue.cpu_task = 1e3

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
            ue.data_size = 1e7
            ue.cpu_task = 1e3
        # uav 初始化
        self.uav1 = UAV.UAV(1, 1, 1, 1500000)
        self.uav2 = UAV.UAV(90, 100, 1, 1500000)
        self.uav3 = UAV.UAV(90, 10, 1, 1500000)
        # self.uav_cluster = [self.uav1]
        # self.uav_cluster = [self.uav2]
        self.uav_cluster = [self.uav3]
        # self.uav_cluster = [self.uav1, self.uav2]
        # bs 初始化
        self.bs1 = BS.BS(10, 75, 50)
        self.bs2 = BS.BS(110, 85, 50)
        self.bs_cluster = [self.bs1, self.bs2]
        # reset 时，要重新绘制无人机飞行轨迹，因此需要重置记录无人机飞行轨迹的数组
        self.uav_cluster_x = []
        self.uav_cluster_y = []
        self.uav_cluster_z = []
        # # 绘制起飞段图形
        # # 按照当前实现的代码逻辑，如果一开始无人机处于一个较低的高度如 1m ，则进行路径规划时很容易就撞到地面，预计要经过很长时间的训练才可以完成起飞路线的规划
        # # 因此在进行训练的时候，一开始就通过调用 launching() 方法使无人机处于一个较高高度如 90m ，在该情况下进行的路径规划不会太过于容易就撞到地面
        for uav in self.uav_cluster:
            self.uav_cluster_x.append([uav.x])
            self.uav_cluster_y.append([uav.y])
            self.uav_cluster_z.append([uav.z])
        # 起飞
        self.uav_cluster = launching(self)
        self.uav3 = self.uav_cluster[0]
        self.situation_n = ["" for m in range(len(self.uav_cluster))]
        self.is_terminal_n = [False for k in range(len(self.uav_cluster))]
        self.uav_pop_count = 0
        # # modified 2 此处在仅将 ue 作为智能体时需要保留，其余场景需要注释掉
        # self.is_terminal_for_only_ue = [True for k in range(len(self.ue_cluster))]
        # self.step_count_for_only_ue = 0

    def _get_obs(self, agent):
        self.state = []
        if isinstance(agent, UAV.UAV):
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
            ue.data_size = 1e7
            ue.cpu_task = 1e3
        # reset 时，将 uav、bs 的 data_size、cpu_task 初始化为0，因为在卸载过程中 uav、bs 上也可能有了一定的任务量
        for uav in self.uav_cluster:
            uav.data_size = 0
            uav.cpu_task = 0
        for bs in self.bs_cluster:
            bs.data_size = 0
            bs.cpu_task = 0

    def step(self, agents, action_n):
        reward_n = []
        delay_n = []
        situation_n = ["" for m in range(len(self.uav_cluster))]
        is_terminal_n = [False for k in range(len(self.uav_cluster))]
        # modified 2
        for agent_index in range(0, 2):
        # for agent_index in range(0, 1):
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
                    reward, delay = self.step_for_ue(action_n, uav_index, ue_index)
                    reward_n.append(reward)
                    delay_n.append(delay)
                    # # modified 2 此处在仅将 ue 作为智能体时需要保留，其余场景需要注释掉
                    # self.step_count_for_only_ue += 1
                    # if self.step_count_for_only_ue == 500:
                    #     return reward_n, self.is_terminal_for_only_ue, delay_n, situation_n

        self.reset2()  # 重置 ue、uav、bs 任务

        return reward_n, is_terminal_n, delay_n, situation_n

    def step_for_ue(self, action_n, uav_index, ue_index):
        # 将 one-hot 形式的 ue 动作 action 转变为单一数值，方便输入到 get_cluster_target_slice_for_maddpg() 方法中进行处理
        agent_index = uav_index + ue_index + 1
        action_transformed = []
        array_temp = action_n[agent_index].tolist()
        max_value = max(array_temp)
        max_index = array_temp.index(max_value)
        action_value = max_index / len(action_n[agent_index]) + 0.5 / len(action_n[agent_index])
        action_transformed.append(action_value)

        delay = get_cluster_target_slice_for_maddpg([self.ue_cluster[ue_index]], self.uav_cluster, self.bs_cluster, self.building_cluster, action_transformed)  # 计算 delay
        # 合理设计 reward 与 delay 的联系，以帮助算法合理收敛
        # reward_n[reward_index] = reward_n[reward_index] + 1000 / (delay - 3)
        # reward_n[reward_index] = reward_n[reward_index] + 100 / (delay - 5)
        reward = - delay * 10
        # reward_n[reward_index] = -delay

        return reward, delay

    def step_for_uav(self, action_n, agent_index):
        reward_n = [0 for m in range(len(self.uav_cluster))]
        # x, y = cal_centroid([[8, 55], [50, 25], [110, 85], [10, 75]])
        # presupposition_point = Agent.Agent(x, y, 30)  # 预设点，即 2ue 与 2bs 组成四边形的形心上空
        # 根据模拟退火算法，在 2ue-1uav-2bs 的场景（其中 ue 和 bs 均为第一和第二个）下，比较理想的预设点应该为(18.93845586, 52.73285563, 18.18901499)
        presupposition_point = Agent.Agent(18.93845586, 52.73285563, 18.18901499)  # 预设点

        uav_index = -1
        for uav in self.uav_cluster:
            uav_index += 1
            if uav_index == agent_index:
                theta, fai, velocity = 0, 0, 0
                if uav_index == 0:
                #     theta = action_n[uav_index][0] * math.pi * 2 + 0.5 * math.pi  # 仰角
                #     fai = action_n[uav_index][1] * math.pi * 2 + 0.25 * math.pi  # 水平夹角
                #     velocity = (action_n[uav_index][2]+0) * 6  # 飞行速度
                # if uav_index == 1:
                #     theta = action_n[uav_index][0] * math.pi * 2 + 0.5 * math.pi  # 仰角
                #     fai = action_n[uav_index][1] * math.pi * 2 - 0.75 * math.pi  # 水平夹角
                #     velocity = (action_n[uav_index][2]+0) * 6  # 飞行速度
                # if uav_index == 2:
                    theta = action_n[uav_index][0] * math.pi * 2 + 0.5 * math.pi  # 仰角
                    fai = action_n[uav_index][1] * math.pi * 2 + 0.75 * math.pi  # 水平夹角
                    velocity = (action_n[uav_index][2]+0) * 6  # 飞行速度
                # velocity = 0  # 飞行速度
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
                    uav.energy = uav.energy - get_power_cost_per_meter(abs(velocity)) * abs(velocity) * 1 - 2000

                uav_uav0_distance = get_uav_uav0_distance(uav, presupposition_point)
                if uav_uav0_distance <= 150:
                    reward_n[uav_index] = reward_n[uav_index] + (150 - uav_uav0_distance) * 10
                    if get_uav_uav0_distance(uav, presupposition_point) <= 90:
                        reward_n[uav_index] = reward_n[uav_index] + (150 - uav_uav0_distance) * 15
                        if get_uav_uav0_distance(uav, presupposition_point) <= 30:
                            reward_n[uav_index] = reward_n[uav_index] + (150 - uav_uav0_distance) * 20
                            if get_uav_uav0_distance(uav, presupposition_point) <= 10:
                                reward_n[uav_index] = reward_n[uav_index] + (150 - uav_uav0_distance) * 30
                else:
                    reward_n[uav_index] = reward_n[uav_index] - 100000  # 给 reward 额外减去一个偏置值
                    # 终止当前 step & episode
                    # print("go_away")
                    # is_terminal = True
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "go_away"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n

                # 坠机（无人机与建筑物相撞或无人机之间相撞）
                if uav_crash_judgement(self.uav_cluster, self.building_cluster):
                    reward_n[uav_index] = reward_n[uav_index] - 100000  # 给 reward 额外减去一个偏置值
                    # 终止当前 step & episode
                    # print("crash")
                    # is_terminal = True
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "crash"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n
                # 坠机（越过 1m 降落高度标准线时速度不为 0，可能撞向地面）
                if uav.z < 1:
                    reward_n[uav_index] = reward_n[uav_index] - 100000  # 给 reward 额外减去一个偏置值
                    # print("down")
                    # 终止当前 step & episode
                    # is_terminal = True
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "down"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n
                # 坠机（未降落时能量即将耗尽，已不足以支撑降落）
                if uav.z >= 1 and uav.energy <= get_power_cost_per_meter(3) * (uav.z-1):
                    reward_n[uav_index] = reward_n[uav_index] - 100000  # 给 reward 额外减去一个偏置值
                    # print("no_energy")
                    # 终止当前 step & episode
                    # is_terminal = True
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "no_energy"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n
                # 若进行原地降落不会与建筑物相撞，可以进入降落程序
                if uav.z >= 1 and (not uav_crash_judgement([uav0_xy], self.building_cluster)):
                    # 已经降落，且上一个step()中 uav 高度大于 1，说明是未经过降落程序自动达到降落高度
                    if uav.z == 1 and uav_z_temp > 1:
                        if uav.energy <= 1500000*0.3:  # 剩余能量较少
                            # 如果此时处于降落区域内给予额外奖励，否则给予额外惩罚
                            if get_uav_uav0_distance(uav, uav0_z) <= 10:
                                reward_n[uav_index] = reward_n[uav_index] + 30000  # 给 reward 额外加上一个偏置值
                                # print("#############################################################################################")
                                # print("well_landing")
                                self.situation_n[uav_index] = "well_landing"
                            else:
                                reward_n[uav_index] = reward_n[uav_index]  # 给 reward 额外减去一个偏置值
                                self.situation_n[uav_index] = "landing"
                        else:  # 剩余能量充足
                            # 如果此时处于降落区域内给予较少的额外奖励，否则给予较多的额外惩罚
                            if get_uav_uav0_distance(uav, uav0_z) <= 10:
                                reward_n[uav_index] = reward_n[uav_index] + 30000  # 给 reward 额外加上一个偏置值
                            else:
                                reward_n[uav_index] = reward_n[uav_index]   # 给 reward 额外减去一个偏置值
                            self.situation_n[uav_index] = "landing"
                        self.uav_pop_count += 1  # 当前 uav 已着陆，uav_pop_count 计数 + 1
                        self.is_terminal_n[uav_index] = True
                        if self.uav_pop_count == len(self.uav_cluster):  # 全部 uav 都已着陆
                            # 终止当前 step & episode
                            # is_terminal = True
                            self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                            return reward_n[uav_index], self.is_terminal_n, self.situation_n
                    # 已经降落，且上一个step()中 uav 高度等于 1，说明该 uav 是停在原地，不需要奖惩
                    if uav.z == 1 and uav_z_temp == 1:
                        self.uav_pop_count += 1  # 当前 uav 已着陆，uav_pop_count 计数 + 1
                        self.is_terminal_n[uav_index] = True
                        self.situation_n[uav_index] = "landing"
                        if self.uav_pop_count == len(self.uav_cluster):  # 全部 uav 都已着陆
                            # 终止当前 step & episode
                            # is_terminal = True
                            self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                            return reward_n[uav_index], self.is_terminal_n, self.situation_n
                    # 暂未降落
                    if uav.z > 1:
                        if uav.energy <= 2 * get_power_cost_per_meter(3) * (uav.z-1):  # 能量即将耗尽(不足以 3m/s 原地降落所需能量的 2 倍)
                            # 如果此时处于降落区域内给予额外奖励，否则给予额外惩罚
                            if get_uav_uav0_distance(uav, uav0_z) <= 5:
                                reward_n[uav_index] = reward_n[uav_index] + 30000  # 给 reward 额外加上一个偏置值
                                # print("#############################################################################################")
                                # print("well_landing")
                                self.situation_n[uav_index] = "well_landing"
                            else:
                                reward_n[uav_index] = reward_n[uav_index] - 100 * get_uav_uav0_distance(uav, uav0)  # 给 reward 额外减去一个偏置值
                                self.situation_n[uav_index] = "landing"
                            landing(self, uav, uav_index)
                            self.uav_pop_count += 1  # 当前 uav 已着陆，uav_pop_count 计数 + 1
                            self.is_terminal_n[uav_index] = True
                            if self.uav_pop_count == len(self.uav_cluster):  # 全部 uav 都已着陆
                                # 终止当前 step & episode
                                # is_terminal = True
                                self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                                return reward_n[uav_index], self.is_terminal_n, self.situation_n
                        else:  # 能量尚且充足，继续迭代寻找降落机会
                            pass
                else:  # 若进行原地降落会与建筑物相撞，继续迭代寻找降落机会
                    pass
                if uav.z > 120 :
                    reward_n[uav_index] = reward_n[uav_index] - 100000  # 给 reward 额外减去一个偏置值
                    # 终止当前 step & episode
                    # print("too_high")
                    # is_terminal = True
                    self.is_terminal_n = [True for m in range(len(self.uav_cluster))]
                    self.situation_n = ["" for m in range(len(self.uav_cluster))]
                    self.situation_n[uav_index] = "too_high"
                    return reward_n[uav_index], self.is_terminal_n, self.situation_n

                if uav.z < 70 and uav.z < uav_z_temp:  # uav 高度过低且继续向下飞
                    reward_n[uav_index] = reward_n[uav_index]   # reward 不变
                if uav.energy < 1500000*0.6 and get_uav_uav0_distance(uav, uav0) >= uav_uav0_distance_temp:  # uav 能量不足且继续远离降落点
                     reward_n[uav_index] = reward_n[uav_index] - 1.3*(get_uav_uav0_distance(uav, uav0))  # 给 reward 额外减去一个偏置值
                if uav.energy >= 1500000*0.6 and uav.z > 1 \
                         and get_clusters_distance([uav], [presupposition_point]) < uav_presupposition_point_distance_temp:  # uav 能量充足，在空中执行任务并靠近预设点位置
                     reward_n[uav_index] = reward_n[uav_index] + 13000/get_uav_uav0_distance(uav, uav0)  # 给 reward 额外加上一个偏置值

                self.uav_cluster_x[uav_index].append(uav.x)
                self.uav_cluster_y[uav_index].append(uav.y)
                self.uav_cluster_z[uav_index].append(uav.z)

        return reward_n[uav_index], self.is_terminal_n, self.situation_n
