import pickle
import matplotlib.pyplot as plt

F = open(r'exp_rewards-20230321214226.pkl', 'rb')

content = pickle.load(F)

print(content)

bx = plt.subplot(1, 1, 1)  # 设置2D绘图空间
# 设置x轴坐标
xx = range(1, 6000, 10)
# 设置y轴坐标
yy = content
bx.plot(xx, yy)  # 绘制对应连线的二维线性图

plt.grid()
plt.show()
