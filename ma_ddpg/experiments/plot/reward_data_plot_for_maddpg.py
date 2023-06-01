import xlrd2
import matplotlib.pyplot as plt
from scipy import signal

book = xlrd2.open_workbook("reward_data-20230531174715.xls")
sheet = book.sheet_by_index(0)

i = 1
ep_reward_list = []

for j in range(1, sheet.nrows):
    if sheet.row_values(j)[i] < -100000:
        ep_reward = sheet.row_values(j-1)[i]
    elif sheet.row_values(j)[i] > 1000000:
        ep_reward = sheet.row_values(j-1)[i]
    else:
        ep_reward = sheet.row_values(j)[i]
    ep_reward_list.append(ep_reward)

ep_reward_list_smooth = signal.savgol_filter(ep_reward_list, 53, 3)
# plt.semilogx(ep_reward_list, label='原始曲线')
# plt.semilogx(ep_reward_list_smooth, label='拟合曲线', color='red')
plt.plot(ep_reward_list, label='原始曲线')
plt.plot(ep_reward_list_smooth, label='拟合曲线', color='red')
plt.legend()  # 显示 label 标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示 label 标签中的中文
plt.rcParams['axes.unicode_minus'] = False  # 显示 label 标签中的负号

# plt.plot(ep_reward_list)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
