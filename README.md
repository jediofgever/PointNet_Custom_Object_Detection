## 3D detecion of custom objects using PointNet

The main code is from <a href="https://github.com/charlesq34/pointnet" target="_blank">PointNet GitHub Repo<a>
Main aim of this projects is to show case how to use PointNet for custom object detection. 

### Dataset

Bored of seeing same datasets all over the place ? , seriously I am, I encourge you to collect your own data and train on it, only then it can feel real. 
So this project uses real data captured from Intel RealSense D435 depth camera. 
Tricky part is; Pointnet can accept number of points as power of 1024. 
So depending on the speed and accuracy that best works for the appliction you aim, 
number of points should be NX1024.

#### If you want to collect your own data{
from a depth camera or anything that provides pointcloud data, I write a detailed tutorial [here](PREPARE_DATA.md) on how to collect, pre-process, label data and finally make it ready for PointNet to consume.
}
#### if you want to use real data I prepared {

}


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



