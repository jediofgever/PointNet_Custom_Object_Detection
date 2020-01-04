
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

