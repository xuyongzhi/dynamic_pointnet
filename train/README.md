# The training pipeline

## Urgent items to be improved
- [ ] Use offline sampled data. Currently, the factor that makes training slow is sampling and grouping operation. In order to avoid this, I will use offline sampled ready data to train. The pre-sample block size is the smallest: 0.02 m. Each block has a block indice. While training, larger block can be got by calculating block indice directly instead of sorting points.
- [ ] Use block instead of ball.
- [ ] Import Matterport3D dataset



## Object detection training plan

### Architecture of detection
#### 1. points sampling  
The KITTI dataset is highly non-uniform point cloud. To evenly cover the whole set, the centroids are selected among input point set by a farthest point sampling (FPS) algorithm.   
Are there other ways to do sampling?

#### 2. points grouping
After getting the smapled point, using the kNN or Ball Query to group point around sampled centriod point.  
Except kNN or Ball Query, are there other methods to do grouping? like cube query.  
The MSG and MRG are also devised to concatenate the features from different scales to handle non-uniform point cloud.  
Trying the MSG firstly.

#### 3. pointNet
The basic unit to process the points, feels like it like the convolutional kernel of CNN, you can use it to build the complex architecture.

#### 4. classification with fully connected network
For classification, two fully connected network are needed to produce classification information, maybe deeper FCNs are needed, for classifications usually need more abstract information.

#### 5. regression with fully connected network
Fore regression, one or two FCNs are needed, bounding box regression is essentially a linear process, thus too abstract information is not helpful. Features for classification and regression should be at different level.

### parameters to tune in the model.py
* N: the number of sampled point, which is very critical
* L: how many set abstraction layers are 
* k: the K value in every layer
* R: what is the radius in every ball
* S: how many scales in MSG
* F: the number of fully connected layers

### parameters to tune in the config.py
* NEGATIVE_CEN_DIST and POSITIVE_CEN_DIST, they affect a lot about number of positive and negative samples
* RPN_BATCHSIZE decides the ratio between positive labels and negative labels
* A: the number of anchors
* P: the number of mini-batch, how many proposals in every mini-batch
* O: overlaping ratio of positive samples and negative samples

### How to do experiments
set #1: (radius=0.3, 0.6, 1.0, 1.8),(nsample= 32, 16, 16, 8), thresh=0.7, acc_38 = 0.1612, acc_30=0.1892  
set #2: (radius=0.3, 0.6, 1.0, 1.8, 2.2), (nsample = 32, 16, 16, 8, 8), thresh=0.6, batchsize = 5

### Some issues
* the unblanced ratio of positive and negative samples.   
Solutions: 1. to find the more positive samples by leasing the positive_ratio; 2. to using the Focal loss function. 3. Reduce the batch size. 3. to increase the anchor number is a very good solution.   
In the last layer, change the bird-view maps into voxel, like image, is useful to find the real proposals, because the points only appear on the surface of object, so it is hard to predict real box with the surface point compared to real central point.  
the dataset only include round 5000 dataset, so it is easy for network to overfit this small dataset if you train network from scratch. The data augmentation is important to train your network, data augmentation methods:1, perturbation randomly; 2, global scaling; 3, global rotation.


