# Description on the utili function in this folder
## KITTI dataset utili function
* `kitti_data_prep.py` is mainly used to generate the h5f file which can be fed to train neural network. The inputs to kitti_data_prep.py is labels and point cloud raw dataset, and output is the h5f files which contain both labels and point cloud.
* `kitti_data_net_provider.py` is class used to read h5f file for training, and provide the interface between raw data and network

### How to use them
1. Changing the dataset path in `kitti_data_prep.py` to generate h5f file.
2. Calling `kitti_data_net_provider.py` to read point cloud and labels in generated h5f file, and pass it to neural network.
3. 
