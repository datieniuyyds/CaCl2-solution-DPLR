import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt
u = mda.Universe("traj.pdb", "traj-unwrap.xtc")
ca_atoms = u.select_atoms('name Ca')
water_atoms = u.select_atoms('name O')
all_dist=[]
for ts in u.trajectory:
    dist_arr = distances.distance_array(ca_atoms.positions, water_atoms.positions, box=u.dimensions)
    all_dist.append(dist_arr)

from collections import defaultdict

frame_data = {}
for frame, distances_frame in enumerate(all_dist):
    count_frames = defaultdict(int)
    for row_num, row in enumerate(distances_frame):
        satisfying_counts = np.sum(row <= 2.375)
        count_frames[satisfying_counts] += 1
    frame_data[frame] = count_frames


number_percentages_all_frames = {}
for frame in range(4001):
    frame_counts = frame_data[frame] if frame in frame_data else {}  # 获取特定帧的数据
    total_counts = sum(frame_counts.values())  # 计算该帧总共出现的数字的数量
    number_percentages = {}
    for number, count in frame_counts.items():
        number_percentages[number] = count / total_counts
    number_percentages_all_frames[frame] = number_percentages

plt.figure()
all_numbers = set()

for frame, number_percentages in number_percentages_all_frames.items():
    numbers = list(number_percentages.keys())
    percentages = list(number_percentages.values())
    all_numbers.update(numbers)

# 将全局数字集合转换为列表，并按照数字排序
sorted_numbers = sorted(all_numbers)

# 遍历每个数字
for number in sorted_numbers:
    number_percentages_list = []
    for frame, number_percentages in number_percentages_all_frames.items():
        if number in number_percentages:
            number_percentages_list.append(number_percentages[number])
        else:
            number_percentages_list.append(0)
    plt.plot(list(number_percentages_all_frames.keys()), number_percentages_list, label=f'{number}')

plt.xlabel('Frame')
plt.ylabel('Percentage')
plt.title('Percentage of Different Number of O Bonded to Ca in Each Frame')
plt.legend()
plt.text(3500, 0.75, '1:24', fontsize=12, color='black')
plt.show()