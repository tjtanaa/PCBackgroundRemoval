if [ ! -d ./build ]; then
  mkdir build
fi

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc src/point_in_polygon.cu -o build/point_in_polygon.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/point_in_polygon.cpp build/point_in_polygon.cu.o -o build/point_in_polygon.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
