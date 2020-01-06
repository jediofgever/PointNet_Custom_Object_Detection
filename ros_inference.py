import os
import sys
from model import *
import numpy as np
import open3d as o3d
import copy
import rospy
import tensorflow as tf
import socket
 
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
 
BATCH_SIZE = 1
BATCH_SIZE_EVAL = 1
NUM_POINT = 4096
BASE_LEARNING_RATE = 0.001
GPU_INDEX = 0
MOMENTUM = 0.9
OPTIMIZER = 'adam'
DECAY_STEP = 300000
DECAY_RATE = 0.5

LOG_DIR = 'log'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

MAX_NUM_POINT = 4096
NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''

    points_ren = []
    lim = points.shape[0]
    print(points.shape)
    for k in range(lim):
        x = points[k,0]
        y = points[k,1]
        z = points[k,2]
        r = int(colors[k,0])
        g = int(colors[k,1])
        b = int(colors[k,2])
        a = int(255)
         
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        pt = [x, y, z, rgb]
        points_ren.append(pt)

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1),
            ]

    header = Header()
    header.frame_id = "camera_link"
    msg = pc2.create_cloud(header, fields, points_ren) 
    return msg

class PointNet_Ros_Node:
  def __init__(self):
    '''initiliaze  ros stuff '''
    self.cloud_pub = rospy.Publisher("output/pointnet/segmented",PointCloud2)
    self.cloud_sub = rospy.Subscriber("pointnet_inference_cloud", PointCloud2,self.callback,queue_size=1)

    is_training = False
    batch = tf.Variable(0)
    learning_rate = get_learning_rate(batch)

    with tf.device('/gpu:'+str(GPU_INDEX)):
      pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
      is_training_pl = tf.placeholder(tf.bool, shape=())

      pred = get_model(pointclouds_pl, is_training_pl)
      loss = get_loss(pred, labels_pl)
      pred_softmax = tf.nn.softmax(pred)
  
      saver = tf.train.Saver()
      
    config = tf.ConfigProto(device_count = {'GPU': 0})
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    self.sess = tf.Session(config=config)
    tf.summary.scalar("loss", loss) 
    merged = tf.summary.merge_all()
    
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=batch)

    self.ops = {'pointclouds_pl': pointclouds_pl,
    'labels_pl': labels_pl,
    'is_training_pl': is_training_pl,
    'pred': pred,
    'loss': loss,
    'train_op': train_op,
    'merged': merged,
    'step': batch}
    MODEL_PATH = "/home/atas/catkin_build_ws/src/ROS_NNs_FANUC_LRMATE200ID/dgcnn/tensorflow/sem_seg/log/model.ckpt"
    # Restore variables from disk.
    saver.restore(self.sess, MODEL_PATH)
    print("Model restored.")
      
  def callback(self, ros_point_cloud):
    xyz = np.array([[0,0,0]])
    rgb = np.array([[0,0,0]])
    #self.lock.acquire()
    gen = pc2.read_points(ros_point_cloud, skip_nans=True)
    int_data = list(gen)

    for x in int_data:
        test = x[3] 
        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f' ,test)
        i = struct.unpack('>l',s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x000000FF)
        # prints r,g,b values in the 0-255 range
                    # x,y,z can be retrieved from the x[0],x[1],x[2]
        xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
        rgb = np.append(rgb,[[r,g,b]], axis = 0)
    self.eval_one_epoch(self.sess, self.ops,xyz,rgb)
   
  def eval_one_epoch(self,sess, ops, xyz, rgb):
    is_training = False
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

    xmax = 3.0
    xmin = -3.0
    current_data = np.zeros((4096,6))
    current_data[:,0:3]  = (xyz[0:NUM_POINT,:]- xmin) / (xmax  - xmin )
    current_data[:,3:6]  = rgb[0:NUM_POINT,:]/(255*255)

    current_data = current_data.reshape(1,4096, 6)
    current_label = np.zeros((1,4096))

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE_EVAL 

    for batch_idx in range(num_batches):
      start_idx = batch_idx * BATCH_SIZE_EVAL
      end_idx = (batch_idx+1) * BATCH_SIZE_EVAL

      feed_dict = {ops['pointclouds_pl']: current_data[:, :],
                    ops['labels_pl']: current_label[:],
                    ops['is_training_pl']: is_training}
      summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                    feed_dict=feed_dict)
      
      pred_label = np.argmax(pred_val, 2) # BxN
      
      test_writer.add_summary(summary, step)
      pred_val = np.argmax(pred_val, 2)
      correct = np.sum(pred_val == current_label[start_idx:end_idx])
      class_color = [[0,255,0],[0,0,255]]
      print(start_idx, end_idx)
      
      for i in range(start_idx, end_idx):
          print(pred_label.shape)
          pred = pred_label[i-start_idx, :]
                    
          pts = current_data[i-start_idx, :, :]
          l = current_label[i-start_idx,:]

          xyz = np.zeros((NUM_POINT, 3))
          colors = np.zeros((NUM_POINT, 3))
          
          for j in range(NUM_POINT):
              l = int(current_label[i, j])

              pred_l = pred_val[i-start_idx, j]
              color = class_color[pred_l]
              color_gt = class_color[l]
              xyz[j, 0] ,a =  pts[j,0] *(xmax  - xmin ) + xmin,pts[j,0] *(xmax  - xmin ) + xmin
              xyz[j, 1] ,b =  pts[j,1] *(xmax  - xmin ) + xmin,pts[j,1] *(xmax  - xmin ) + xmin
              xyz[j, 2] ,c =  pts[j,2] *(xmax  - xmin ) + xmin,pts[j,2] *(xmax  - xmin ) + xmin
              xyz[j, 0] =  c
              xyz[j, 1] = -a
              xyz[j, 2] = -b
 
              colors[j,0],colors[j,1],colors[j,2] = color[0], color[1], color[2]


          #out_pcd = o3d.geometry.PointCloud()    
          #out_pcd.points = o3d.utility.Vector3dVector(xyz)
          #out_pcd.colors = o3d.utility.Vector3dVector(colors)
          #o3d.io.write_point_cloud("/home/atas/predition.ply",out_pcd)         
          msg = xyzrgb_array_to_pointcloud2(xyz,colors,rospy.Time.now(),"camera_link", 1)
          self.cloud_pub.publish(msg)
          

  
if __name__=='__main__':
  
  ic = PointNet_Ros_Node()
  rospy.init_node('ros_point_cloud',anonymous=True)
  rospy.spin()
