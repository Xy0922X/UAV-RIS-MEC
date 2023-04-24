""" 用户设备类 """


from agents.Agent import Agent


class UE(Agent):
    # cpu算力
    cpu_power = 10 ** 1

    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)
