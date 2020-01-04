import numpy as np 
import open3d 
import open3d as o3d
import random
 
#dummycomment

from plyfile import PlyData, PlyElement
import h5py
f = h5py.File('real_data.h5','r')

def rotate_point_cloud(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                  [sinval, cosval, 0],
                  [-0, 0, 1]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data

print(f.keys())
data = f['data']
label = f['label']

true_data = np.zeros((len(data),len(data[0]),len(data[0][0])))
augmented_data = np.zeros((len(data),len(data[0]),len(data[0][0])))
true_data = data
true_labels = label

f_scale_down_xyz = h5py.File('f_scale_down_xyz.h5','w')
f_scale_up_xyz = h5py.File('f_scale_up_xyz.h5','w')
f_light_up_colors = h5py.File('f_light_up_colors.h5','w')
f_light_down_colors = h5py.File('f_light_down_colors.h5','w')
f_random_rotate = h5py.File('f_random_rotate.h5','w')

print(len(data))
for i in range (0, len(data)):
	#scale down xyz
	augmented_data[i,:,0:3] = random.uniform(0.7, 1.0) * true_data[i,:,0:3]
	augmented_data[i,:,3:6] = true_data[i,:,3:6]
f_scale_down_xyz.create_dataset('data', data = augmented_data)
f_scale_down_xyz.create_dataset('label', data = true_labels)

for i in range (0, len(data)):
	#scale up xyz
	augmented_data[i,:,0:3] = random.uniform(1.0, 1.3) * true_data[i,:,0:3]
	augmented_data[i,:,3:6] =  true_data[i,:,3:6]
f_scale_up_xyz.create_dataset('data', data = augmented_data)
f_scale_up_xyz.create_dataset('label', data = true_labels)

for i in range (0, len(data)):
	#change lighting conditions(up)
	augmented_data[i,:,0:3] =  true_data[i,:,0:3]
	augmented_data[i,:,3:6] = random.uniform(1.0, 1.15) * true_data[i,:,3:6]
f_light_up_colors.create_dataset('data', data = augmented_data)
f_light_up_colors.create_dataset('label', data = true_labels)

for i in range (0, len(data)):
	#change lighting conditions(down)
	augmented_data[i,:,0:3] =  true_data[i,:,0:3]
	augmented_data[i,:,3:6] = random.uniform(0.85, 1.0) * true_data[i,:,3:6]
f_light_down_colors.create_dataset('data', data = augmented_data)
f_light_down_colors.create_dataset('label', data = true_labels)

rotated_data = rotate_point_cloud(true_data[:,:,0:3])
augmented_data[:,:,0:3] =  rotated_data
augmented_data[:,:,3:6] =  true_data[:,:,3:6]
f_random_rotate.create_dataset('data', data = augmented_data)
f_random_rotate.create_dataset('label', data = true_labels)









