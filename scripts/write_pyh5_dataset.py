import h5py
import os, os.path
import numpy as np
from plyfile import PlyData, PlyElement
import glob
import open3d as o3d


# path joining version for other paths
DIR = '/home/atas/pcl_dataset'

 
for m  in range(0,4):

    data = np.zeros((2048, 4096, 9), dtype = np.float32)
    label = np.zeros((2048,  4096),dtype = np.uint8)
 
	
    f = h5py.File('data'+str(m)+'.h5', 'w')
    
    for i in range(0, 2048):
        plydata = PlyData.read("/home/atas/pcl_dataset/" + str(m*2048+i) + ".ply")
        print("done ",m*2048+i ,"files ")
        for j in range(0, 4096):
                    ## NORMALIZE POINT  CLOUD 
            MAX_X, MAX_, MAX_Z = 4,4,4
            MIN_X, MIN_Y, MIN_Z = -4,-4,-4
            data[i, j] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j], plydata['vertex']['red'][j]/255.0, plydata['vertex']['green'][j]/255.0, 
                                  plydata['vertex']['blue'][j]/255.0,(plydata['vertex']['x'][j]-MIN_X)/8.0,(plydata['vertex']['y'][j]-MIN_Y)/8.0,(plydata['vertex']['z'][j]-MIN_Z)/8.0]
            
            if(plydata['vertex']['label'][j] == 1):
                plydata['vertex']['blue'][j]= 255
                label[i,j] = 1   
            else:
                label[i,j] = 0   
        #plydata.write(str(m*2048+i) + ".ply")


    f.create_dataset('data', data = data)
    f.create_dataset('label', data = label)





