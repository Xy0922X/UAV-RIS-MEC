""" 建筑物类 """


class IRS:
    # 左侧定义点位置坐标
    xyz = [x, y, z] = [0, 0, 0]
    # 右侧定义点相对左侧定义点的位置坐标变化值
    dxyz = [dx, dy, dz] = [0, 0, 0]

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # 输出类变量
    def say(self):
        print(self.x, self.y, self.z)
