""" 基站类 """


from agents.Agent import Agent


class BS(Agent):
    # cpu算力
    cpu_power = 10 ** 3

    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)


class BS_Local(Agent):
    # cpu算力
    cpu_power = 5 ** 3

    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)