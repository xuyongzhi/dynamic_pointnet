#/bin/bash
#/usr/local/cuda-8.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++-4.9 -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0


/usr/local/cuda-8.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /home/y/tf1.2.1-p2/local/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /home/y/tf1.3-p2/local/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0


#/apps/cuda/8.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /apps/tensorflow/1.3.1-python2.7/lib/python2.7/site-packages/tensorflow/include -I /apps/cuda/8.0/include -lcudart -L /apps/cuda/8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
