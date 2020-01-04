import h5py
import os, os.path
import numpy as np
from plyfile import PlyData, PlyElement

 

test_pcd_frame = np.zeros((4096,10),dtype=float)

plydata = PlyData.read("/home/atas/scripts/1053.ply")

for j in range(0, 4096):


    label = 0
    if(plydata['vertex']['blue'][j] > 0 or plydata['vertex']['red'][j] > 0 or plydata['vertex']['green'][j] > 0 ):
        label = 1

    else:
        label = 0
                        

    MAX_X, MAX_, MAX_Z = 4,4,4
    MIN_X, MIN_Y, MIN_Z = -4,-4,-4
 
    test_pcd_frame[j] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j], 
                  plydata['vertex']['red'][j]/255.0, plydata['vertex']['green'][j]/255.0,plydata['vertex']['blue'][j]/255.0, 
                 (plydata['vertex']['x'][j]-MIN_X)/8.0,(plydata['vertex']['y'][j]-MIN_Y)/8.0,(plydata['vertex']['z'][j]-MIN_Z)/8.0,label]

 

np.save('1053.npy',test_pcd_frame)

print(test_pcd_frame.shape)
