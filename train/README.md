# The training pipeline

## Urgent items to be improved
- [ ] Use offline sampled data. Currently, the factor that makes training slow is sampling and grouping operation. In order to avoid this, I will use offline sampled ready data to train. The pre-sample block size is the smallest: 0.02 m. Each block has a block indice. While training, larger block can be got by calculating block indice directly instead of sorting points.
- [ ] Use block instead of ball.
- [ ] Import Matterport3D dataset



## Object detection training plan

### 1. architecture of detection
#### 2. points sampling
#### 3. points grouping
#### 4. pointNet
#### 5. classification with fully connected network
#### 6. regression with fully connected network


### parameters to tune
* how many set abstraction layers
* the K value in every layer
* what is the radius in every ball
