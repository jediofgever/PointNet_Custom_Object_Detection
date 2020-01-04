#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <pwd.h>
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace boost::filesystem;

int
main (int argc, char** argv)
{


 // pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);
  pcl::PCLPointCloud2 cloud;
  pcl::PointCloud<pcl::PointXYZL>::Ptr vertices( new pcl::PointCloud<pcl::PointXYZL> );
  pcl::PCDReader reader;

  passwd* pw = getpwuid(getuid());
  std::string home_path(pw->pw_dir);
 

 for (directory_entry& entry : directory_iterator(home_path+"/POINTNET_LABELED_REAL_DATASET/")){

    const boost::filesystem::path path = entry.path();
    std::string path_string =  path.string();
    std::string org_path = path_string;
    std::cout << path_string << '\n';
    size_t i = 0; 

    for ( ; i < path_string.length(); i++ ){ if ( isdigit(path_string[i]) ) break; }

    // remove the first chars, which aren't digits
    path_string = path_string.substr(i, path_string.length() - i );

    // convert the remaining text to an integer
    int id = atoi(path_string.c_str());
    std::cout  << id << std::endl;

  if (reader.read(org_path, cloud, 0) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file %s \n",org_path.c_str());
    continue;
  }
  pcl::fromPCLPointCloud2(cloud, *vertices ); 

    pcl::PLYWriter writer;
    writer.write(home_path+"/POINTNET_LABELED_REAL_DATASET/"+std::to_string(id)+".ply", cloud, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), true, false);

}

  return (0);
}
