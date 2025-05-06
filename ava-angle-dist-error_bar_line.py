import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from MDAnalysis.lib import distances
import matplotlib.pyplot as plt
import time
u = mda.Universe("traj.pdb")
ca_atoms = u.select_atoms('name Ca')
water_atoms = u.select_atoms('name O')
cl_atoms = u.select_atoms('name Cl')
#water_indices=water_atoms.indices+1
from MDAnalysis.lib.distances import calc_angles
import MDAnalysis.analysis.distances as mda_dist
from MDAnalysis.core.topologyobjects import Angle
h_atoms = u.select_atoms('name H')
from pylab import mpl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
bins=np.linspace(0,40,num=100+1)
angle_bins = np.linspace(0, 180,num=50)
x_values = np.linspace(bins[0], bins[-1], num=bins.shape[0])
y_values = np.linspace(angle_bins[0], angle_bins[-1], num=angle_bins.shape[0])
average_probability = np.zeros((len(x_values), len(y_values)))
def pbc_distance(vec, box):
    vec = np.array(vec)
    half_box = box / 2.0
    return vec - (vec / box).astype(int) * box - (vec > half_box) * box + (vec < -half_box) * box

def angle_bisector(posA, posB, posC, box):
    posA = np.array(posA)
    posB = np.array(posB)
    posC = np.array(posC)
    box = np.array(box)
    vecBA = pbc_distance(posA - posB, box)
    vecBC = pbc_distance(posC - posB, box)
    normBA = vecBA / np.linalg.norm(vecBA)
    normBC = vecBC / np.linalg.norm(vecBC)
    bisector = normBA + normBC
    bisector /= np.linalg.norm(bisector)
    return bisector

def calc_angles_by_vec(repeated_vectors, vector_OCa):
    angles = []
    for vec1, vec2 in zip(repeated_vectors, vector_OCa):
        dot_product = np.dot(vec1, vec2)
        norm_vector1 = np.linalg.norm(vec1)
        norm_vector2 = np.linalg.norm(vec2)
        cosine_angle = dot_product / (norm_vector1 * norm_vector2)
        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)
        angles.append(angle_degrees)
    return np.array(angles)

# angles = distances.calc_angles(ca_atoms.positions, water_atoms.positions[0], bisector + posB)
# print (ca_atoms.positions, water_atoms.positions)
# angles_deg = np.degrees(angles)
# angles_deg

average_y_values_list = []
for ts in u.trajectory[1000:4001]:
    box = ts.dimensions[:3] # Box dimensions
    hydrogen_oxygen_distance= distances.distance_array(water_atoms.positions, h_atoms.positions,box=u.dimensions)
    h_index=[]
    water_indices=[]
    for i, ho_dist in enumerate(hydrogen_oxygen_distance):
        connected_h_atoms_indices = np.where(ho_dist <= 1.24)[0]
        if len(connected_h_atoms_indices) == 2:
            h_index.append(connected_h_atoms_indices)
            water_indices.append(i)
        else:
            print(f"error: {(h_atoms.indices)[connected_h_atoms_indices]},{ts.frame}")
    first_h=[item[0] for item in h_index]
    second_h=[item[1] for item in h_index]
    selected_water_atoms_indices = np.array(water_indices)
    posA = h_atoms.positions[first_h]
    posB = water_atoms.positions[selected_water_atoms_indices]
    posC = h_atoms.positions[second_h] # Change this as needed
    posD = ca_atoms.positions
    bisector = angle_bisector(posA, posB, posC, box)
    #print("Angle bisector vector:", bisector, bisector + posB)
    vector_Ob=bisector
    vectors_ca_o = []
    for ca_coord in posD:
        vectors_ca_o_for_ca = []
        for oxygen_coord in posB:
            vector_cao = ca_coord - oxygen_coord
            vectors_ca_o_for_ca.append(vector_cao)
        vectors_ca_o.append(vectors_ca_o_for_ca)
    vector_OCa = np.array(vectors_ca_o)
    #print(vector_OCa)
    #vector_OCa = vector_OCa.reshape(9,)
    vector_OCa = pbc_distance(vector_OCa, box)
    vector_OCa=vector_OCa.reshape((len(ca_atoms.indices)*len(selected_water_atoms_indices)),3)
    repeated_vectors = np.tile(vector_Ob, (len(ca_atoms.indices), 1))
    #print(vector_OCa)
    #print(repeated_vectors)
    angles_Ob_OCa = calc_angles_by_vec(repeated_vectors, vector_OCa)
    ca_oxygen_distance= distances.distance_array(posD, posB,box=u.dimensions)
    ca_oxygen_distance_flat = ca_oxygen_distance.flatten()
    angles_Ob_OCa_flat=angles_Ob_OCa.flatten()
    ca_o_dis_angle = np.column_stack((ca_oxygen_distance_flat, angles_Ob_OCa_flat))
    #print(ca_o_dis_angle)
    average_y_values = []
    for i, x in enumerate(x_values):
        idx = np.where((ca_o_dis_angle[:, 0] >= x) & (ca_o_dis_angle[:, 0] < x + (bins[1] - bins[0])))[0]
        y_values_in_range = (ca_o_dis_angle[:, 1])[idx]
        #print(y_values_in_range.shape)
        hist, _ = np.histogram(y_values_in_range, range=[0, 180], bins=angle_bins.shape[0])
        normalized_hist = hist / np.sum(hist)
        average_probability[i, :] += normalized_hist
        if len(y_values_in_range) > 0:  # 确保该 bin 内有数据点
            average_y_values.append(np.mean(y_values_in_range))
        else:
            average_y_values.append(np.nan)
    average_y_values_list.append(average_y_values)
#print(average_y_values_list[0])
#print(average_y_values_list[1])
average_probability /= len(u.trajectory[1000:4001])
x_values_rounded = np.round(x_values, 2)
y_values_rounded = np.round(y_values, 2)
average_y_values_array = np.array(average_y_values_list)
average_y_values_mean = np.nanmean(average_y_values_array, axis=0)
average_y_values_std = np.nanstd(average_y_values_array, axis=0)
data = pd.DataFrame({'x_values': np.repeat(x_values_rounded, len(y_values)),
                     'y_values': np.tile(y_values_rounded, len(x_values)),
                     'probability': average_probability.flatten()})
data.to_csv('heatmap_data_ion_o_mid_angle.txt', sep='\t', index=False)
data2 = np.column_stack((x_values, average_y_values_std, average_y_values_mean))
np.savetxt('ava-angle-dist-error_bar_data.txt', data2, fmt='%.8f', delimiter='\t', header='x_values\taverage_y_values_std\taverage_y_values_mean', comments='')
#print(average_y_values_mean)
plt.figure(figsize=(8, 6))
f = data.pivot(index='y_values', columns='x_values', values='probability')
# Heatmap
ax1 = plt.gca()
mask=f==0
sns.heatmap(f, mask=mask, annot=False, cmap='viridis', fmt='.2f', ax=ax1)
plt.xlabel('d$_{Ca-O}$ (Å)')
plt.ylabel('Angle (°)')
plt.title('Heatmap with Average Probability')
x_value_index = np.abs(x_values - 3.4).argmin()
plt.axvline(x=x_value_index, color='red', linestyle='--') 
plt.show()

plt.plot(x_values, average_y_values_mean)
plt.fill_between(x_values, average_y_values_mean - average_y_values_std, average_y_values_mean + average_y_values_std,
                 color='lightblue', label='Standard Deviation', alpha=0.5)
plt.xlabel('d$_{Ca-O}$ (Å)')
plt.ylabel('Average angle (°)') 
plt.title('Average angle vs O-Ca-distance')
#x_value_index = np.abs(x_values - 3.4).argmin()
plt.axvline(x=3.4, color='red', linestyle='--')
plt.show()