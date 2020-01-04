## 3D detecion of custom objects using PointNet

The main code is from <a href="https://github.com/charlesq34/pointnet" target="_blank">PointNet GitHub Repo<a>
Main aim of this projects is to show case how to use PointNet for custom object detection. 

### Dataset
This project uses real data captured from Intel RealSense D435 depth camera. 
Tricky part is; Pointnet can accept number of points as power of 1024. 
So depending on the speed and accuracy that best works for the appliction you aim, 
number of points should be NX1024.

#### pre-process data and create data samples
In this project we take N to be 4, so point clouds are pre-processed to have exactly 4096 points.
This is achieved using PCL's utilities. First strategy is removing ground plane. 

```cpp
void PointCloudManager::removeGroundPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered) {
    pcl::PointIndicesPtr ground(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_filtered);
    cloud_filtered->header = cloud->header;
    cloud_filtered->height = 1;
    cloud_filtered->width = cloud_filtered->points.size();  
}
```

This will remove ground plane, next thing we can check number of points and if still far from Nx4096 points we can down sample cloud with;

```cpp 
void PointCloudManager::downsamplePCL(pcl::PCLPointCloud2::Ptr cloud, pcl::PCLPointCloud2::Ptr cloud_filtered) {
    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.008f, 0.008f, 0.008f);
    sor.filter(*cloud_filtered);
}
```
The leafsize parameter needs to be adjusted such that it will get as close as possible to N*4096
finally after this we can trim remaining few points and save cloud as .pcd file ;

```cpp
while (downsampled_cloud_PointCloud->points.size() > 4096) {
    downsampled_cloud_PointCloud->points.pop_back();
}
std::string pcd_path  = "/path/to/datset/"+str(counter)+".pcd"
savePCDFileASCII(pcd_path, *downsampled_cloud_PointCloud);
counter++;
```
I saved my pcd files to a folder named, POINTNET_REAL_DATA. later this directory will be again used.


#### label, pre-processed data
Now that we have .pcd files and we shall label the objects, 
I found out that https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor
to be very easy to use and fast to label clouds, please refer to link about how to use, the labeling tool,it is a browser based one, I labeled around 60 frames in 1.5 hour. So it is actually pretty fast. After finising labeling process, the files should b again saved as .pcd files. I saved them under a directory named POINTNET_LABELED_REAL_DATA. 

Unfourtunaltely the labeling tool cuts out RGB values in cloud also the labeling cannot be read properly using readPCD, therefore we need to get back RGB values from the original .pcd files as well as read labels properly. To do this tasks I have write some scripts. First we convert .pcd files to ply files in order to properly read labels

in scripts folder do ;

```cpp
rm -rf build
mkdir build
cd build
cmake .. 
make 
./convert_pcd2ply
```

.ply files with same names(names should be integers! 0.pcd, 1.pcd..) will be dumped in to /home/user_name/POINTNET_LABELED_REAL_DATA
#### create .h5 files of real data
At this stage we have .ply files which includes X,y,z,label information under POINTNET_LABELED_REAL_DATA,
to add r,g,b values we can access to original data under  POINTNET_REAL_DATA, put all X,Y,Z,R,G,B,LABEL values together and create .h5 file.

the script for doing this is ; 
Note, depending on number of frames you need adjust, NUM_FRAMES in this file

```cpp
python3 write_pyh5_real.py
```
this will create a .h5 file named real_data.h5

### augment data
60 frames of data is not enough to achieve a solid performance
therfore we need to enhance the data, 
random scale down , scale up , changing rotation of point cloud randomly helps
a data augentation script is provided under scripts, 

```cpp
python3 augment_real_data.py
```
there will be 5 additional .h5 files starting with f , after this step we are finally all set to start training


### Training
Training script is a ipython notebook.
After augmentation step we should have 5 files starting with f and real_dataset with .h5 extensions.
to iterate through this files conviently put them under a directory, which you need to adjust in train.ipynb, for example I renemad the final files to d0,d1...d5
and the name of directory is FULL_H5_DATA.

Also remeber to adjust number of frames to the same number that you set in write_pyh5_real.py 

```cpp
ipython notebook 
```
direct to train.ipynb, 

and run each cell, make sure you dont recieve an error,
According to number of epochs the training will hold on. 

### Testing
Testing is again done in ipython notebook. 


### Selected Projects that Use PointNet

* <a href="http://stanford.edu/~rqi/pointnet2/" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017) A hierarchical feature learning framework on point clouds. The PointNet++ architecture applies PointNet recursively on a nested partitioning of the input point set. It also proposes novel layers for point clouds with non-uniform densities.
* <a href="http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w13/Engelmann_Exploring_Spatial_Context_ICCV_2017_paper.pdf" target="_blank">Exploring Spatial Context for 3D Semantic Segmentation of Point Clouds</a> by Engelmann et al. (ICCV 2017 workshop). This work extends PointNet for large-scale scene segmentation.
* <a href="https://arxiv.org/abs/1710.04954" target="_blank">PCPNET: Learning Local Shape Properties from Raw Point Clouds</a> by Guerrero et al. (arXiv). The work adapts PointNet for local geometric properties (e.g. normal and curvature) estimation in noisy point clouds.
* <a href="https://arxiv.org/abs/1711.06396" target="_blank">VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection</a> by Zhou et al. from Apple (arXiv) This work studies 3D object detection using LiDAR point clouds. It splits space into voxels, use PointNet to learn local voxel features and then use 3D CNN for region proposal, object classification and 3D bounding box estimation.
* <a href="https://arxiv.org/abs/1711.08488" target="_blank">Frustum PointNets for 3D Object Detection from RGB-D Data</a> by Qi et al. (arXiv) A novel framework for 3D object detection with RGB-D data. The method proposed has achieved first place on KITTI 3D object detection benchmark on all categories (last checked on 11/30/2017).



