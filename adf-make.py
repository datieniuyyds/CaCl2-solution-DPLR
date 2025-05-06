import numpy as np
import matplotlib.pyplot as plt
import math
data = np.loadtxt('data-outter-3141.adf',comments=['#','4000000'])
angle=data[:,1]
distribution=data[:,2]
sinangle=distribution*np.sin(3.14*angle/180)
y=sinangle
x=angle
np.save('adf-ffmd',np.transpose([x,y]))
adf=np.load('adf-ffmd.npy')
np.savetxt('adf-ffmd.txt',adf)
plt.plot(x,y,label='1:6-DPMD')
plt.xlabel('angle')
plt.ylabel('Distribution')
plt.legend(loc="upper right")
plt.title("Angle-Distribution")
plt.show()
