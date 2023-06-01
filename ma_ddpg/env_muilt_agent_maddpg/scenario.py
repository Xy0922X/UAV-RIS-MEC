import matplotlib.pyplot as plt
import numpy as np
import math

from ma_ddpg.multiagent_envs.env_functions import line_of_sight_judgement

ue1 = (1700, 1700, 0)
ue2 = (2000, 1500, 0)
ue3 = (1400, 1650, 0)
ue4 = (3800, 3500, 0)
ue5 = (4200, 2000, 0)
ue_cluster = [ue1, ue2, ue3, ue4, ue5]

uav = (1000, 1500, 10)
uav_cluster = [uav]

irs1 = (2000, 1800, 29)
irs_cluster = [irs1]

bs1 = (1600, 1750, 50)
bs2 = (3400, 3200, 50)
bs3 = (3400, 3600, 50)
bs4 = (4400, 2100, 50)
bs5 = (4800, 1600, 50)
bs_cluster = [bs1, bs2, bs3, bs4, bs5]

building7 = (3600, 3000, 0, 260, 140, 20)
building8 = (1300, 1400, 0, 260, 140, 20)
building9 = (1000, 1800, 0, 260, 140, 20)
building10 = (1000, 1300, 0, 260, 140, 20)
building1 = (1800, 1800, 0, 260, 140, 25)
building2 = (1300, 1800, 0, 260, 140, 20)
building3 = (1600, 1400, 0, 260, 140, 20)
building4 = (4200, 1800, 0, 260, 140, 20)
building5 = (4600, 1900, 0, 260, 140, 20)
building6 = (3600, 3600, 0, 260, 140, 20)
building11 = (3600, 3300, 0, 260, 140, 20)
building12 = (3000, 3000, 0, 260, 140, 20)
building13 = (3000, 3300, 0, 260, 140, 20)
building14 = (3000, 3600, 0, 260, 140, 20)

# 其他住房建筑1
building15 = (3000, 1500, 0, 80, 60, 40)
building16 = (3200, 1500, 0, 80, 60, 40)
building17 = (3400, 1500, 0, 80, 60, 40)
building18 = (3600, 1500, 0, 80, 60, 40)
building19 = (3800, 1500, 0, 80, 60, 40)
building20 = (4000, 1500, 0, 80, 60, 40)

building21 = (3000, 1700, 0, 80, 60, 40)
building22 = (3200, 1700, 0, 80, 60, 40)
building23 = (3400, 1700, 0, 80, 60, 40)
building24 = (3600, 1700, 0, 80, 60, 40)
building25 = (3800, 1700, 0, 80, 60, 40)
building26 = (4000, 1700, 0, 80, 60, 40)

building27 = (3000, 1900, 0, 80, 60, 40)
building28 = (3200, 1900, 0, 80, 60, 40)
building29 = (3400, 1900, 0, 80, 60, 40)
building30 = (3600, 1900, 0, 80, 60, 40)
building31 = (3800, 1900, 0, 80, 60, 40)
building32 = (4000, 1900, 0, 80, 60, 40)

# 其他住房建筑2
building33 = (3000, 2100, 0, 80, 60, 40)
building34 = (3200, 2100, 0, 80, 60, 40)
building35 = (3400, 2100, 0, 80, 60, 40)
building36 = (3600, 2100, 0, 80, 60, 40)
building37 = (3800, 2100, 0, 80, 60, 40)
building38 = (4000, 2100, 0, 80, 60, 40)

building39 = (3000, 2300, 0, 80, 60, 40)
building40 = (3200, 2300, 0, 80, 60, 40)
building41 = (3400, 2300, 0, 80, 60, 40)
building42 = (3600, 2300, 0, 80, 60, 40)
building43 = (3800, 2300, 0, 80, 60, 40)
building44 = (4000, 2300, 0, 80, 60, 40)

building45 = (3000, 2500, 0, 80, 60, 40)
building46 = (3200, 2500, 0, 80, 60, 40)
building47 = (3400, 2500, 0, 80, 60, 40)
building48 = (3600, 2500, 0, 80, 60, 40)
building49 = (3800, 2500, 0, 80, 60, 40)
building50 = (4000, 2500, 0, 80, 60, 40)

# 其他住房建筑3
building51 = (3000, 1300, 0, 80, 60, 40)
building52 = (3200, 1300, 0, 80, 60, 40)
building53 = (3400, 1300, 0, 80, 60, 40)
building54 = (3600, 1300, 0, 80, 60, 40)
building55 = (3800, 1300, 0, 80, 60, 40)
building56 = (4000, 1300, 0, 80, 60, 40)

building57 = (3000, 1100, 0, 80, 60, 40)
building58 = (3200, 1100, 0, 80, 60, 40)
building59 = (3400, 1100, 0, 80, 60, 40)
building60 = (3600, 1100, 0, 80, 60, 40)
building61 = (3800, 1100, 0, 80, 60, 40)
building62 = (4000, 1100, 0, 80, 60, 40)

building63 = (3000, 900, 0, 80, 60, 40)
building64 = (3200, 900, 0, 80, 60, 40)
building65 = (3400, 900, 0, 80, 60, 40)
building66 = (3600, 900, 0, 80, 60, 40)
building67 = (3800, 900, 0, 80, 60, 40)
building68 = (4000, 900, 0, 80, 60, 40)

# 其他住房建筑4
building69 = (3000, 700, 0, 80, 60, 40)
building70 = (3200, 700, 0, 80, 60, 40)
building71 = (3400, 700, 0, 80, 60, 40)
building72 = (3600, 700, 0, 80, 60, 40)
building73 = (3800, 700, 0, 80, 60, 40)
building74 = (4000, 700, 0, 80, 60, 40)

building75 = (3000, 500, 0, 80, 60, 40)
building76 = (3200, 500, 0, 80, 60, 40)
building77 = (3400, 500, 0, 80, 60, 40)
building78 = (3600, 500, 0, 80, 60, 40)
building79 = (3800, 500, 0, 80, 60, 40)
building80 = (4000, 500, 0, 80, 60, 40)

building81 = (3000, 300, 0, 80, 60, 40)
building82 = (3200, 300, 0, 80, 60, 40)
building83 = (3400, 300, 0, 80, 60, 40)
building84 = (3600, 300, 0, 80, 60, 40)
building85 = (3800, 300, 0, 80, 60, 40)
building86 = (4000, 300, 0, 80, 60, 40)

building87 = (3000, 100, 0, 80, 60, 40)
building88 = (3200, 100, 0, 80, 60, 40)
building89 = (3400, 100, 0, 80, 60, 40)
building90 = (3600, 100, 0, 80, 60, 40)
building91 = (3800, 100, 0, 80, 60, 40)
building92 = (4000, 100, 0, 80, 60, 40)

# 其他住房建筑5
building93 = (0, 700, 0, 80, 60, 40)
building94 = (200, 700, 0, 80, 60, 40)
building95 = (400, 700, 0, 80, 60, 40)
building96 = (600, 700, 0, 80, 60, 40)
building97 = (800, 700, 0, 80, 60, 40)
building98 = (1000, 700, 0, 80, 60, 40)

building99 = (0, 500, 0, 80, 60, 40)
building100 = (200, 500, 0, 80, 60, 40)
building101 = (400, 500, 0, 80, 60, 40)
building102 = (600, 500, 0, 80, 60, 40)
building103 = (800, 500, 0, 80, 60, 40)
building104 = (1000, 500, 0, 80, 60, 40)

building105 = (0, 300, 0, 80, 60, 40)
building106 = (200, 300, 0, 80, 60, 40)
building107 = (400, 300, 0, 80, 60, 40)
building108 = (600, 300, 0, 80, 60, 40)
building109 = (800, 300, 0, 80, 60, 40)
building110 = (1000, 300, 0, 80, 60, 40)

building111 = (0, 100, 0, 80, 60, 40)
building112 = (200, 100, 0, 80, 60, 40)
building113 = (400, 100, 0, 80, 60, 40)
building114 = (600, 100, 0, 80, 60, 40)
building115 = (800, 100, 0, 80, 60, 40)
building116 = (1000, 100, 0, 80, 60, 40)

building117 = (0, 900, 0, 80, 60, 40)
building118 = (200, 900, 0, 80, 60, 40)
building119 = (400, 900, 0, 80, 60, 40)
building120 = (600, 900, 0, 80, 60, 40)
building121 = (800, 900, 0, 80, 60, 40)
building122 = (1000, 900, 0, 80, 60, 40)

# 其他住房建筑6
building123 = (1200, 700, 0, 80, 60, 40)
building124 = (1400, 700, 0, 80, 60, 40)
building125 = (1600, 700, 0, 80, 60, 40)
building126 = (1800, 700, 0, 80, 60, 40)
building127 = (2000, 700, 0, 80, 60, 40)
building128 = (2200, 700, 0, 80, 60, 40)
building129 = (2400, 700, 0, 80, 60, 40)
building130 = (2600, 700, 0, 80, 60, 40)
building131 = (2800, 700, 0, 80, 60, 40)
building132 = (3000, 700, 0, 80, 60, 40)

building133 = (1200, 500, 0, 80, 60, 40)
building134 = (1400, 500, 0, 80, 60, 40)
building135 = (1600, 500, 0, 80, 60, 40)
building136 = (1800, 500, 0, 80, 60, 40)
building137 = (2000, 500, 0, 80, 60, 40)
building138 = (2200, 500, 0, 80, 60, 40)
building139 = (2400, 500, 0, 80, 60, 40)
building140 = (2600, 500, 0, 80, 60, 40)
building141 = (2800, 500, 0, 80, 60, 40)
building142 = (3000, 500, 0, 80, 60, 40)

building143 = (1200, 300, 0, 80, 60, 40)
building144 = (1400, 300, 0, 80, 60, 40)
building145 = (1600, 300, 0, 80, 60, 40)
building146 = (1800, 300, 0, 80, 60, 40)
building147 = (2000, 300, 0, 80, 60, 40)
building148 = (2200, 300, 0, 80, 60, 40)
building149 = (2400, 300, 0, 80, 60, 40)
building150 = (2600, 300, 0, 80, 60, 40)
building151 = (2800, 300, 0, 80, 60, 40)
building152 = (3000, 300, 0, 80, 60, 40)

building153 = (1200, 100, 0, 80, 60, 40)
building154 = (1400, 100, 0, 80, 60, 40)
building155 = (1600, 100, 0, 80, 60, 40)
building156 = (1800, 100, 0, 80, 60, 40)
building157 = (2000, 100, 0, 80, 60, 40)
building158 = (2200, 100, 0, 80, 60, 40)
building159 = (2400, 100, 0, 80, 60, 40)
building160 = (2600, 100, 0, 80, 60, 40)
building161 = (2800, 100, 0, 80, 60, 40)
building162 = (3000, 100, 0, 80, 60, 40)

building163 = (1200, 100, 0, 80, 60, 40)
building164 = (1400, 100, 0, 80, 60, 40)
building165 = (1600, 100, 0, 80, 60, 40)
building166 = (1800, 100, 0, 80, 60, 40)
building167 = (2000, 100, 0, 80, 60, 40)
building168 = (2200, 100, 0, 80, 60, 40)
building169 = (2400, 100, 0, 80, 60, 40)
building170 = (2600, 100, 0, 80, 60, 40)
building171 = (2800, 100, 0, 80, 60, 40)
building172 = (3000, 100, 0, 80, 60, 40)

building173 = (1200, 900, 0, 80, 60, 40)
building174 = (1400, 900, 0, 80, 60, 40)
building175 = (1600, 900, 0, 80, 60, 40)
building176 = (1800, 900, 0, 80, 60, 40)
building177 = (2000, 900, 0, 80, 60, 40)
building178 = (2200, 900, 0, 80, 60, 40)
building179 = (2400, 900, 0, 80, 60, 40)
building180 = (2600, 900, 0, 80, 60, 40)
building181 = (2800, 900, 0, 80, 60, 40)
building182 = (3000, 900, 0, 80, 60, 40)

building183 = (1200, 3900, 0, 80, 60, 40)
building184 = (1400, 3900, 0, 80, 60, 40)
building185 = (1600, 3900, 0, 80, 60, 40)
building186 = (1800, 3900, 0, 80, 60, 40)
building187 = (2000, 3900, 0, 80, 60, 40)
building188 = (2200, 3900, 0, 80, 60, 40)
building189 = (2400, 3900, 0, 80, 60, 40)
building190 = (2600, 3900, 0, 80, 60, 40)
building191 = (2800, 3900, 0, 80, 60, 40)
building192 = (3000, 3900, 0, 80, 60, 40)

building193 = (1200, 3700, 0, 80, 60, 40)
building194 = (1400, 3700, 0, 80, 60, 40)
building195 = (1600, 3700, 0, 80, 60, 40)
building196 = (1800, 3700, 0, 80, 60, 40)
building197 = (2000, 3700, 0, 80, 60, 40)
building198 = (2200, 3700, 0, 80, 60, 40)
building199 = (2400, 3700, 0, 80, 60, 40)
building200 = (2600, 3700, 0, 80, 60, 40)
building201 = (2800, 3700, 0, 80, 60, 40)
building202 = (3000, 3700, 0, 80, 60, 40)

building203 = (1200, 3500, 0, 80, 60, 40)
building204 = (1400, 3500, 0, 80, 60, 40)
building205 = (1600, 3500, 0, 80, 60, 40)
building206 = (1800, 3500, 0, 80, 60, 40)
building207 = (2000, 3500, 0, 80, 60, 40)
building208 = (2200, 3500, 0, 80, 60, 40)
building209 = (2400, 3500, 0, 80, 60, 40)
building210 = (2600, 3500, 0, 80, 60, 40)
building211 = (2800, 3500, 0, 80, 60, 40)
building212 = (3000, 3500, 0, 80, 60, 40)

building213 = (1200, 3300, 0, 80, 60, 40)
building214 = (1400, 3300, 0, 80, 60, 40)
building215 = (1600, 3300, 0, 80, 60, 40)
building216 = (1800, 3300, 0, 80, 60, 40)
building217 = (2000, 3300, 0, 80, 60, 40)
building218 = (2200, 3300, 0, 80, 60, 40)
building219 = (2400, 3300, 0, 80, 60, 40)
building220 = (2600, 3300, 0, 80, 60, 40)
building221 = (2800, 3300, 0, 80, 60, 40)
building222 = (3000, 3300, 0, 80, 60, 40)

building223 = (1200, 3100, 0, 80, 60, 40)
building224 = (1400, 3100, 0, 80, 60, 40)
building225 = (1600, 3100, 0, 80, 60, 40)
building226 = (1800, 3100, 0, 80, 60, 40)
building227 = (2000, 3100, 0, 80, 60, 40)
building228 = (2200, 3100, 0, 80, 60, 40)
building229 = (2400, 3100, 0, 80, 60, 40)
building230 = (2600, 3100, 0, 80, 60, 40)
building231 = (2800, 3100, 0, 80, 60, 40)
building232 = (3000, 3100, 0, 80, 60, 40)

building233 = (100, 3000, 0, 260, 140, 20)
building233 = (500, 3300, 0, 260, 140, 20)
building234 = (100, 3300, 0, 260, 140, 20)
building235 = (500, 3300, 0, 260, 140, 20)
building236 = (100, 3600, 0, 260, 140, 20)
building237 = (500, 3600, 0, 260, 140, 20)
building238 = (100, 3900, 0, 260, 140, 20)
building239 = (500, 3900, 0, 260, 140, 20)



building_cluster = [building1, building2, building3, building4, building5, building6, building7, building8, building9,
                    building10,
                    building11, building12, building13, building14, building15, building16, building17, building18,
                    building19, building20,
                    building21, building22, building23, building24, building25, building26, building27, building28,
                    building29, building30,
                    building31, building32, building33, building34, building35, building36, building37, building38,
                    building39, building40,
                    building41, building42, building43, building44, building45, building46, building47, building48,
                    building49, building50,
                    building51, building52, building53, building54, building55, building56, building57, building58,
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
                    building151, building152, building153, building154, building155, building156, building157,
                    building158, building159, building160,
                    building161, building162, building163, building164, building165, building166, building167,
                    building168, building169, building170,
                    building171, building172, building173, building174, building175, building176, building177,
                    building178, building179, building180,
                    building181, building182, building183, building184, building185, building186, building187,
                    building188, building189, building190,
                    building191, building192, building193, building194, building195, building196, building197,
                    building198, building199, building200,
                    building201, building202, building203, building204, building205, building206, building207,
                    building208, building209, building210,
                    building211, building212, building213, building214, building215, building216, building217,
                    building218, building219, building220,
                    building221, building222, building223, building224, building225, building226, building227,
                    building228, building229, building230,
                    building231, building232, building233, building234, building235, building236, building237,
                    building238, building239,
                    ]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题

x_irs = []
y_irs = []
z_irs = []
x_ue = []
y_ue = []
z_ue = []
x_bs = []
y_bs = []
z_bs = []
x_bu = []
y_bu = []
z_bu = []
dx_bu = []
dy_bu = []
dz_bu = []

for ue in ue_cluster:
    x_ue.append(ue[0])
    y_ue.append(ue[1])
    z_ue.append(ue[2])

for irs in irs_cluster:
    x_irs.append(irs[0])
    y_irs.append(irs[1])
    z_irs.append(irs[2])

for bs in bs_cluster:
    x_bs.append(bs[0])
    y_bs.append(bs[1])
    z_bs.append(bs[2])

for building in building_cluster:
    x_bu.append(building[0])
    y_bu.append(building[1])
    z_bu.append(building[2])
    dx_bu.append(building[3])
    dy_bu.append(building[4])
    dz_bu.append(building[5])

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
for bs in bs_cluster:
    x = [bs[0], bs[0], bs[0] - math.sqrt(3) * 2, bs[0] + math.sqrt(3) * 2]
    y = [bs[1], bs[1] + 3, bs[1] - 3, bs[1] - 3]
    z = [bs[2], 0, 0, 0]
    for i in range(4):
        for j in range(4):
            ax1.plot((x[i], x[j]), (y[i], y[j]), (z[i], z[j]), color="cadetblue", alpha=0.5)

ax1.plot(400, 0, 0, c='white')
ax1.plot(0, 600, 0, c='white')
ax1.plot(0, 0, 1, c='white')
ax1.plot(0, 0, 0, c='white')
ax1.bar3d(x_bu, y_bu, z_bu, dx_bu, dy_bu, dz_bu, color='lightgray')
ax1.scatter3D(x_ue, y_ue, z_ue, c='r', s=4)
ax1.scatter3D(x_irs, y_irs, z_irs, 'darkslateblue', marker='^')
plt.xlim(0, 5000)
plt.ylim(0, 4000)
ax1.set_zticks(np.arange(0, 100, 20))
ax1.set_box_aspect([50, 40, 10])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.grid()
plt.show()
