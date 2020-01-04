## 3D detecion of custom objects using PointNet

The main code is from <a href="https://github.com/charlesq34/pointnet" target="_blank">PointNet GitHub Repo<a>
Main aim of this projects is to show case how to use PointNet for custom object detection. 

### Dataset
This project uses real data captured from Intel RealSense D435 depth camera. 
Tricky part is; Pointnet can accept number of points as power of 1024. 
So depending on the speed and accuracy that best works for the appliction you aim, 
number of points should be NX1024.

#### pre-process data
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
    ROS_INFO("x %.2f y %.2f z %.2f c %.2f", coefficients->values[0], coefficients->values[1], coefficients->values[2],
             coefficients->values[3]);    
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
finally after this we can trim reamining few points such as ;

```cpp
while (downsampled_cloud_PointCloud->points.size() > 4096) {
    downsampled_cloud_PointCloud->points.pop_back();
}
```


### Training

Once you have downloaded and prepared data, to start training use main.ipynb. 

### Visualise data

For data visualization you can use vis_data_vispy.py file.

### Selected Projects that Use PointNet

* <a href="http://stanford.edu/~rqi/pointnet2/" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017) A hierarchical feature learning framework on point clouds. The PointNet++ architecture applies PointNet recursively on a nested partitioning of the input point set. It also proposes novel layers for point clouds with non-uniform densities.
* <a href="http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w13/Engelmann_Exploring_Spatial_Context_ICCV_2017_paper.pdf" target="_blank">Exploring Spatial Context for 3D Semantic Segmentation of Point Clouds</a> by Engelmann et al. (ICCV 2017 workshop). This work extends PointNet for large-scale scene segmentation.
* <a href="https://arxiv.org/abs/1710.04954" target="_blank">PCPNET: Learning Local Shape Properties from Raw Point Clouds</a> by Guerrero et al. (arXiv). The work adapts PointNet for local geometric properties (e.g. normal and curvature) estimation in noisy point clouds.
* <a href="https://arxiv.org/abs/1711.06396" target="_blank">VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection</a> by Zhou et al. from Apple (arXiv) This work studies 3D object detection using LiDAR point clouds. It splits space into voxels, use PointNet to learn local voxel features and then use 3D CNN for region proposal, object classification and 3D bounding box estimation.
* <a href="https://arxiv.org/abs/1711.08488" target="_blank">Frustum PointNets for 3D Object Detection from RGB-D Data</a> by Qi et al. (arXiv) A novel framework for 3D object detection with RGB-D data. The method proposed has achieved first place on KITTI 3D object detection benchmark on all categories (last checked on 11/30/2017).



