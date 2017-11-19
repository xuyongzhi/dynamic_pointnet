#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0




g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I  /apps/tensorflow/1.3.1-cudnn6.0-python2.7/lib/python2.7/site-packages/tensorflow/include   -I  /apps/cuda/8.0/include -lcudart -L /apps/cuda/8.0/lib64/  -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I  /apps/tensorflow/1.2.1-cudnn6.0-python2.7/lib/python2.7/site-packages/tensorflow/include   -I  /apps/cuda/8.0/include -lcudart -L /apps/cuda/8.0/lib64/  -O2 -D_GLIBCXX_USE_CXX11_ABI=0

#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I  /apps/tensorflow/1.3.1-python2.7/lib/python2.7/site-packages/tensorflow/include   -I  /apps/cuda/8.0/include -lcudart -L /apps/cuda/8.0/lib64/  -O2 -D_GLIBCXX_USE_CXX11_ABI=0



