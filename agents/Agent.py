""" 智能体类 """


class Agent:
    # 位置坐标
    xyz = [x, y, z] = [0, 0, 0]
    # 数据量
    data_size = 0
    # 任务量（CPU周期数）
    cpu_task = 0
    # cpu算力
    cpu_power = 0
    # 5G带宽，假设为 350 MHz
    bandwidth = 3.5 * 10 ** 8
    # 信号功率
    S = 150000

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)


