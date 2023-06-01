""" 无人机类 """


import math
from agents.Agent import Agent


class UAV(Agent):
    # 能量 500kJ，参考论文为 Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning
    energy = 1500000
    # cpu算力
    cpu_power = 1 * 10 ** 9


    def __init__(self, x, y, z, energy):
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]
        self.energy = energy

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)

    # 飞行（坐标随时间戳变化）
    def flight_trace(self, theta, fai, velocity, time_slot):
        self.x = self.x + velocity * math.sin(theta) * math.cos(fai) * time_slot
        self.y = self.y + velocity * math.sin(theta) * math.sin(fai) * time_slot
        self.z = self.z + velocity * math.cos(theta) * time_slot
        self.xyz = [self.x, self.y, self.z]
