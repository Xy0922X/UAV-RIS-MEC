import xlrd2
from agents import UE, BS, UAV, Building, IRS
from ma_ddpg.experiments.plot import ue_uav_bs_plot

ue1 = UE.UE(140, 165, 0)
ue2 = UE.UE(170, 170, 0)
ue3 = UE.UE(200, 150, 0)
ue4 = UE.UE(380, 350, 0)
ue5 = UE.UE(520, 200, 0)

uav1 = UAV.UAV(0, 200, 1, 1500000)

bs1 = BS.BS_local(160, 175, 20)
bs2 = BS.BS(340, 320, 20)
bs3 = BS.BS(400, 340, 20)
bs4 = BS.BS(540, 210, 20)
bs5 = BS.BS(580, 160, 20)

building1 = Building.Building(100, 150, 0, 26, 14, 20)
building2 = Building.Building(100, 180, 0, 26, 14, 20)
building3 = Building.Building(130, 140, 0, 26, 14, 20)
building4 = Building.Building(130, 180, 0, 26, 14, 20)
building5 = Building.Building(160, 140, 0, 26, 14, 20)
building6 = Building.Building(180, 180, 0, 26, 14, 25)
building7 = Building.Building(300, 300, 0, 26, 14, 20)
building8 = Building.Building(300, 330, 0, 26, 14, 20)
building9 = Building.Building(300, 360, 0, 26, 14, 20)
building10 = Building.Building(360, 300, 0, 26, 14, 20)
building11 = Building.Building(360, 330, 0, 26, 14, 20)
building12 = Building.Building(360, 360, 0, 26, 14, 20)
building13 = Building.Building(520, 180, 0, 26, 14, 20)
building14 = Building.Building(560, 200, 0, 26, 14, 20)

# 其他住房建筑1
building15 = Building.Building(300, 150, 0, 8, 6, 40)
building16 = Building.Building(320, 150, 0, 8, 6, 40)
building17 = Building.Building(340, 150, 0, 8, 6, 40)
building18 = Building.Building(360, 150, 0, 8, 6, 40)
building19 = Building.Building(380, 150, 0, 8, 6, 40)
building20 = Building.Building(400, 150, 0, 8, 6, 40)

building21 = Building.Building(300, 170, 0, 8, 6, 40)
building22 = Building.Building(320, 170, 0, 8, 6, 40)
building23 = Building.Building(340, 170, 0, 8, 6, 40)
building24 = Building.Building(360, 170, 0, 8, 6, 40)
building25 = Building.Building(380, 170, 0, 8, 6, 40)
building26 = Building.Building(400, 170, 0, 8, 6, 40)

building27 = Building.Building(300, 190, 0, 8, 6, 40)
building28 = Building.Building(320, 190, 0, 8, 6, 40)
building29 = Building.Building(340, 190, 0, 8, 6, 40)
building30 = Building.Building(360, 190, 0, 8, 6, 40)
building31 = Building.Building(380, 190, 0, 8, 6, 40)
building32 = Building.Building(400, 190, 0, 8, 6, 40)

# 其他住房建筑2
building33 = Building.Building(300, 210, 0, 8, 6, 40)
building34 = Building.Building(320, 210, 0, 8, 6, 40)
building35 = Building.Building(340, 210, 0, 8, 6, 40)
building36 = Building.Building(360, 210, 0, 8, 6, 40)
building37 = Building.Building(380, 210, 0, 8, 6, 40)
building38 = Building.Building(400, 210, 0, 8, 6, 40)

building39 = Building.Building(300, 230, 0, 8, 6, 40)
building40 = Building.Building(320, 230, 0, 8, 6, 40)
building41 = Building.Building(340, 230, 0, 8, 6, 40)
building42 = Building.Building(360, 230, 0, 8, 6, 40)
building43 = Building.Building(380, 230, 0, 8, 6, 40)
building44 = Building.Building(400, 230, 0, 8, 6, 40)

building45 = Building.Building(300, 250, 0, 8, 6, 40)
building46 = Building.Building(320, 250, 0, 8, 6, 40)
building47 = Building.Building(340, 250, 0, 8, 6, 40)
building48 = Building.Building(360, 250, 0, 8, 6, 40)
building49 = Building.Building(380, 250, 0, 8, 6, 40)
building50 = Building.Building(400, 250, 0, 8, 6, 40)

# 其他住房建筑3
building51 = Building.Building(300, 130, 0, 8, 6, 40)
building52 = Building.Building(320, 130, 0, 8, 6, 40)
building53 = Building.Building(340, 130, 0, 8, 6, 40)
building54 = Building.Building(360, 130, 0, 8, 6, 40)
building55 = Building.Building(380, 130, 0, 8, 6, 40)
building56 = Building.Building(400, 130, 0, 8, 6, 40)

building57 = Building.Building(300, 110, 0, 8, 6, 40)
building58 = Building.Building(320, 110, 0, 8, 6, 40)
building59 = Building.Building(340, 110, 0, 8, 6, 40)
building60 = Building.Building(360, 110, 0, 8, 6, 40)
building61 = Building.Building(380, 110, 0, 8, 6, 40)
building62 = Building.Building(400, 110, 0, 8, 6, 40)

building63 = Building.Building(300, 90, 0, 8, 6, 40)
building64 = Building.Building(320, 90, 0, 8, 6, 40)
building65 = Building.Building(340, 90, 0, 8, 6, 40)
building66 = Building.Building(360, 90, 0, 8, 6, 40)
building67 = Building.Building(380, 90, 0, 8, 6, 40)
building68 = Building.Building(400, 90, 0, 8, 6, 40)

# 其他住房建筑4
building69 = Building.Building(100, 310, 0, 8, 6, 40)
building70 = Building.Building(120, 310, 0, 8, 6, 40)
building71 = Building.Building(140, 310, 0, 8, 6, 40)
building72 = Building.Building(160, 310, 0, 8, 6, 40)
building73 = Building.Building(180, 310, 0, 8, 6, 40)
building74 = Building.Building(200, 310, 0, 8, 6, 40)

building75 = Building.Building(100, 330, 0, 8, 6, 40)
building76 = Building.Building(120, 330, 0, 8, 6, 40)
building77 = Building.Building(140, 330, 0, 8, 6, 40)
building78 = Building.Building(160, 330, 0, 8, 6, 40)
building79 = Building.Building(180, 330, 0, 8, 6, 40)
building80 = Building.Building(200, 330, 0, 8, 6, 40)

# 其他住房建筑5
building81 = Building.Building(100, 350, 0, 8, 6, 40)
building82 = Building.Building(120, 350, 0, 8, 6, 40)
building83 = Building.Building(140, 350, 0, 8, 6, 40)
building84 = Building.Building(160, 350, 0, 8, 6, 40)
building85 = Building.Building(180, 350, 0, 8, 6, 40)
building86 = Building.Building(200, 350, 0, 8, 6, 40)

building87 = Building.Building(100, 370, 0, 8, 6, 40)
building88 = Building.Building(120, 370, 0, 8, 6, 40)
building89 = Building.Building(140, 370, 0, 8, 6, 40)
building90 = Building.Building(160, 370, 0, 8, 6, 40)
building91 = Building.Building(180, 370, 0, 8, 6, 40)
building92 = Building.Building(200, 370, 0, 8, 6, 40)

# 其他住房建筑4
building93 = Building.Building(100, 390, 0, 8, 6, 40)
building94 = Building.Building(120, 390, 0, 8, 6, 40)
building95 = Building.Building(140, 390, 0, 8, 6, 40)
building96 = Building.Building(160, 390, 0, 8, 6, 40)
building97 = Building.Building(180, 390, 0, 8, 6, 40)
building98 = Building.Building(200, 390, 0, 8, 6, 40)

building99 = Building.Building(100, 210, 0, 8, 6, 40)
building100 = Building.Building(120, 210, 0, 8, 6, 40)
building101 = Building.Building(140, 210, 0, 8, 6, 40)
building102 = Building.Building(160, 210, 0, 8, 6, 40)
building103 = Building.Building(180, 210, 0, 8, 6, 40)
building104 = Building.Building(200, 210, 0, 8, 6, 40)

building105 = Building.Building(100, 230, 0, 8, 6, 40)
building106 = Building.Building(120, 230, 0, 8, 6, 40)
building107 = Building.Building(140, 230, 0, 8, 6, 40)
building108 = Building.Building(160, 230, 0, 8, 6, 40)
building109 = Building.Building(180, 230, 0, 8, 6, 40)
building110 = Building.Building(200, 230, 0, 8, 6, 40)

building111 = Building.Building(100, 250, 0, 8, 6, 40)
building112 = Building.Building(120, 250, 0, 8, 6, 40)
building113 = Building.Building(140, 250, 0, 8, 6, 40)
building114 = Building.Building(160, 250, 0, 8, 6, 40)
building115 = Building.Building(180, 250, 0, 8, 6, 40)
building116 = Building.Building(200, 250, 0, 8, 6, 40)

# 环境不用加
building117 = Building.Building(0, 10, 0, 26, 14, 20)
building118 = Building.Building(50, 10, 0, 26, 14, 20)
building119 = Building.Building(100, 10, 0, 26, 14, 20)
building120 = Building.Building(150, 10, 0, 26, 14, 20)
building121 = Building.Building(200, 10, 0, 26, 14, 20)
building122 = Building.Building(250, 10, 0, 26, 14, 20)
building123 = Building.Building(300, 10, 0, 26, 14, 20)
building124 = Building.Building(350, 10, 0, 26, 14, 20)
building125 = Building.Building(400, 10, 0, 26, 14, 20)
building126 = Building.Building(450, 10, 0, 26, 14, 20)
building127 = Building.Building(500, 10, 0, 26, 14, 20)
building128 = Building.Building(550, 10, 0, 26, 14, 20)
building129 = Building.Building(600, 10, 0, 26, 14, 20)

building130 = Building.Building(0, 30, 0, 26, 14, 20)
building154 = Building.Building(50, 30, 0, 26, 14, 20)
building131 = Building.Building(100, 30, 0, 26, 14, 20)
building132 = Building.Building(150, 30, 0, 26, 14, 20)
building133 = Building.Building(200, 30, 0, 26, 14, 20)
building134 = Building.Building(250, 30, 0, 26, 14, 20)
building135 = Building.Building(300, 30, 0, 26, 14, 20)
building136 = Building.Building(350, 30, 0, 26, 14, 20)
building137 = Building.Building(400, 30, 0, 26, 14, 20)
building138 = Building.Building(450, 30, 0, 26, 14, 20)
building139 = Building.Building(500, 30, 0, 26, 14, 20)
building140 = Building.Building(550, 30, 0, 26, 14, 20)
building141 = Building.Building(600, 30, 0, 26, 14, 20)

building142 = Building.Building(0, 50, 0, 26, 14, 20)
building143 = Building.Building(50, 50, 0, 26, 14, 20)
building155 = Building.Building(100, 50, 0, 26, 14, 20)
building144 = Building.Building(150, 50, 0, 26, 14, 20)
building145 = Building.Building(200, 50, 0, 26, 14, 20)
building146 = Building.Building(250, 50, 0, 26, 14, 20)
building147 = Building.Building(300, 50, 0, 26, 14, 20)
building148 = Building.Building(350, 50, 0, 26, 14, 20)
building149 = Building.Building(400, 50, 0, 26, 14, 20)
building150 = Building.Building(450, 50, 0, 26, 14, 20)
building151 = Building.Building(500, 50, 0, 26, 14, 20)
building152 = Building.Building(550, 50, 0, 26, 14, 20)
building153 = Building.Building(600, 50, 0, 26, 14, 20)

ue_cluster = [ue1, ue2, ue3, ue4, ue5]

uav_cluster = [uav1]

bs_cluster = [bs1, bs2, bs3, bs4, bs5]

building_cluster = [building1, building2, building3, building4, building5, building6, building7, building8,
                    building9, building10, building11, building12, building13, building14, building15, building16,
                    building17, building18,
                    building19, building20, building21, building22, building23, building24, building25, building26,
                    building27, building28,
                    building29, building30, building31, building32, building33, building34, building35, building36,
                    building37, building38,
                    building39, building40, building41, building42, building43, building44, building45, building46,
                    building47, building48,
                    building49, building50, building51, building52, building53, building54, building55, building56,
                    building57, building58,
                    building59, building60,
                    building61, building62, building63, building64, building65, building66, building67, building68,
                    building69, building70,
                    building71, building72, building73, building74, building75, building76, building77, building78,
                    building79, building80,
                    building81, building82, building83, building84, building85, building86, building87, building88,
                    building89, building90,
                    building91, building92, building93, building94, building95, building96, building97, building98,
                    building99, building100,
                    building101, building102, building103, building104, building105, building106, building107,
                    building108, building109, building110,
                    building111, building112, building113, building114, building115, building116, building117,
                    building118, building119, building120,
                    building121, building122, building123, building124, building125, building126, building127,
                    building128, building129, building130,
                    building131, building132, building133, building134, building135, building136, building137,
                    building138, building139, building140,
                    building141, building142, building143, building144, building145, building146, building147,
                    building148, building149, building150,
                    building151, building152, building153, building154, building155]

irs1 = IRS.IRS(200, 190, 26)
irs_cluster = [irs1]

# # 静态绘图
# book = xlrd2.open_workbook("episodes_data-20230321215910.xls")
# sheets = []
# # episodes_count 用来描述想要绘制的图像所对应的数据页范围
# for episodes_count in range(114, 115):
#     # offset 用来描述所绘制图像所对应的数据页，实际上此处对应数据页为第 0 + offset*1 或 0 + offset*2、1 + offset*2 页（每次绘制一架无人机的轨迹图像）
#     offset = episodes_count
#     sheets = []
#     # 单无人机数据
#     for i in range(0 + offset * 1, 1 + offset * 1):
#     # # 双无人机数据
#     # for i in range(0 + offset * 2, 2 + offset * 2):
#         sheets.append(book.sheets()[i])
#     uav_cluster_x = [[] for i in range(6)]
#     uav_cluster_y = [[] for i in range(6)]
#     uav_cluster_z = [[] for i in range(6)]
#     time_array = [[] for i in range(6)]
#     target_slice_array = [[] for i in range(6)]
#     ep_reward = []
#     episode = []
#     steps = []
#     sheet_index = -1
#     for sheet in sheets:
#         sheet_index += 1
#         i = 0
#         for j in range(1, sheets[sheet_index].nrows):
#             if sheets[sheet_index].row_values(j)[i] != '':
#                 uav_cluster_x[sheet_index].append(sheets[sheet_index].row_values(j)[i])
#
#         i = 1
#         for j in range(1, sheets[sheet_index].nrows):
#             if sheets[sheet_index].row_values(j)[i] != '':
#                 uav_cluster_y[sheet_index].append(sheets[sheet_index].row_values(j)[i])
#
#         i = 2
#         for j in range(1, sheets[sheet_index].nrows):
#             if sheets[sheet_index].row_values(j)[i] != '':
#                 uav_cluster_z[sheet_index].append(sheets[sheet_index].row_values(j)[i])
#
#         i = 3
#         for j in range(32, sheets[sheet_index].nrows):
#             if sheets[sheet_index].row_values(j)[i] != '':
#                 time_array[sheet_index].append(sheets[sheet_index].row_values(j)[i])
#
#         i = 4
#         if sheet_index == 0 or sheet_index == 1:
#             for j in range(32, sheets[sheet_index].nrows):
#                 if sheets[sheet_index].row_values(j)[i] != '':
#                     target_slice_array[sheet_index].append(float(sheets[sheet_index].row_values(j)[i].strip("[]")) - 0.5)
#                     # target_slice_array[sheet_index].append(0)
#         else:
#             for j in range(32, sheets[sheet_index].nrows):
#                 if sheets[sheet_index].row_values(j)[i] != '':
#                     target_slice_array[sheet_index].append(float(sheets[sheet_index].row_values(j)[i].strip("[]")))
#
#         ep_reward.append(sheets[sheet_index].row_values(1)[5])
#         episode.append(sheets[sheet_index].row_values(1)[6])
#         steps.append(sheets[sheet_index].row_values(1)[7])
#
#     ue_uav_bs_plot.plot_static(uav_cluster_x, uav_cluster_y, uav_cluster_z, ue_cluster, bs_cluster,
#                                time_array, target_slice_array, building_cluster, ep_reward, episode, steps)

# 动态绘图
book = xlrd2.open_workbook("episodes_data-20230531223100.xls")
sheets = []
# episodes_count 用来描述想要绘制的图像所对应的数据页范围
for episodes_count in range(78, 79):
    # offset 用来描述所绘制图像所对应的数据页，实际上此处对应数据页为第 0 + offset*1 或 0 + offset*2、1 + offset*2 页（每次绘制一架无人机的轨迹图像）
    offset = episodes_count
    sheets = []
    # 单无人机数据
    for i in range(0 + offset * 1, 1 + offset * 1):
        sheets.append(book.sheets()[i])
    uav_cluster_x = [[] for i in range(6)]
    uav_cluster_y = [[] for i in range(6)]
    uav_cluster_z = [[] for i in range(6)]
    time_array = [[] for i in range(6)]
    target_slice_array = [[] for i in range(6)]
    ep_reward = []
    episode = []
    steps = []
    sheet_index = -1
    for sheet in sheets:
        sheet_index += 1
        i = 0
        for j in range(1, sheets[sheet_index].nrows):
            if sheets[sheet_index].row_values(j)[i] != '':
                uav_cluster_x[sheet_index].append(sheets[sheet_index].row_values(j)[i])

        i = 1
        for j in range(1, sheets[sheet_index].nrows):
            if sheets[sheet_index].row_values(j)[i] != '':
                uav_cluster_y[sheet_index].append(sheets[sheet_index].row_values(j)[i])

        i = 2
        for j in range(1, sheets[sheet_index].nrows):
            if sheets[sheet_index].row_values(j)[i] != '':
                uav_cluster_z[sheet_index].append(sheets[sheet_index].row_values(j)[i])

        i = 3
        for j in range(22, sheets[sheet_index].nrows):
            if sheets[sheet_index].row_values(j)[i] != '':
                time_array[sheet_index].append(sheets[sheet_index].row_values(j)[i])

        i = 4
        if sheet_index == 0 or sheet_index == 1:
            for j in range(22, sheets[sheet_index].nrows):
                if sheets[sheet_index].row_values(j)[i] != '':
                    cleaned_string = sheets[sheet_index].row_values(j)[i].strip("[]")
                    number_strings = cleaned_string.split(", ")
                    float_numbers = [float(num) for num in number_strings]
                    target_slice_array[sheet_index].append(sum(float_numbers) / len(float_numbers))

        else:
            for j in range(22, sheets[sheet_index].nrows):
                cleaned_string = sheets[sheet_index].row_values(j)[i].strip("[]")
                number_strings = cleaned_string.split(", ")
                float_numbers = [float(num) for num in number_strings]
                target_slice_array[sheet_index].append(sum(float_numbers) / len(float_numbers))

        ep_reward.append(sheets[sheet_index].row_values(1)[5])
        episode.append(sheets[sheet_index].row_values(1)[6])
        steps.append(sheets[sheet_index].row_values(1)[7])

        ue_uav_bs_plot.plot(uav_cluster_x, uav_cluster_y, uav_cluster_z, ue_cluster, bs_cluster, irs_cluster,
                            time_array, target_slice_array, building_cluster, ep_reward, episode, steps)
