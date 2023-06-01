""" 基站类 """

from agents.Agent import Agent


class BS(Agent):
    # cpu算力
    cpu_power = 1 * 10 ** 10

    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)


class BS_local(BS):
    cpu_power = 0.5 * 10 ** 10

    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)

# print(BS_local.cpu_power)
# bs1 = BS_local(1600, 1750, 25)
# print(BS.cpu_power)
