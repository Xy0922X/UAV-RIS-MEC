""" 用户设备类 """


from agents.Agent import Agent


class UE(Agent):
    # cpu算力
    cpu_power = 1 * 10 ** 8
    # 数据量
    data_size = 1e6
    # 任务量（CPU周期数）
    cpu_task = 1e9

    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)



