time_list=[]
for i_step in range(4001):
    time = i_step*0.0005
    time_list.append(time)
import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('hbnum.xvg',comments=['#', '@'])
hbonds = data[:,1]
hbonds_last_2000 = hbonds[-2000:]
time_last_2000 = np.array(time_list[-2000:])
#time = np.array(time_list)
#fig, ax = plt.subplots()
#ax.plot(time, hbonds)
#ax.set_xlabel('Time (ps)')
#ax.set_ylabel('Number of Hydrogen Bonds')
#ax.set_title('Time-Hydrogen Bonds Relationship')
#plt.show()
#x=time
ava_hbonds=np.mean(hbonds_last_2000)
print(ava_hbonds)
#plt.bar(x, y, width=0.8, color='blue', edgecolor='black')
#plt.xlabel('Time (ps)')
#plt.ylabel('Hydrogen bonding count')
#plt.title('Hydrogen bonding count over time')
#plt.show()
#std_dev = np.std(hbonds_last_2000)
#print("Standard Deviation of Hydrogen Bonds (Last 2000 data points):", std_dev)
