""" 建筑物类 """


class Building:
    # 左侧定义点位置坐标
    xyz = [x, y, z] = [0, 0, 0]
    # 右侧定义点相对左侧定义点的位置坐标变化值
    dxyz = [dx, dy, dz] = [0, 0, 0]

    def __init__(self, x, y, z, dx, dy, dz):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.xyz = [x, y, z]
        self.dxyz = [dx, dy, dz]

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z, self.dx, self.dy, self.dz)


class Point:
    # 点位置坐标
    xyz = [x, y, z] = [0, 0, 0]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.xyz = [x, y]

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)
