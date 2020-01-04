import numpy as np 
import open3d 
import open3d as o3d
 
#dummycomment

from plyfile import PlyData, PlyElement
import h5py
f = h5py.File('f_random_rotate.h5','r')

print(f.keys())
data = f['data']
label = f['label']

xyz = np.zeros((len(data[1]), 3))
colors = np.zeros((len(data[1]), 3))

xyz[:, 0] = data[44][:,0]
xyz[:, 1] = data[44][:,1]
xyz[:, 2] = data[44][:,2]
colors[:, 0] = data[44][:,3] * label[44][:]
colors[:, 1] = data[44][:,4] * label[44][:]
colors[:, 2] = data[44][:,5] * label[44][:]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)


o3d.visualization.draw_geometries([pcd])

