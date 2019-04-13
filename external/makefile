TF_LIB= /home/dmri/.local/lib/python3.5/site-packages/tensorflow/
nvcc = /usr/local/cuda-9.0/bin/nvcc
cudalib = /usr/local/cuda-9.0/lib64/
tensorflow = /home/dmri/.local/lib/python3.5/site-packages/tensorflow/include/

all: structural_losses/tf_nndistance_so.so structural_losses/tf_nndistance_g.cu.o structural_losses/tf_approxmatch_so.so structural_losses/tf_approxmatch_g.cu.o
.PHONY : all

structural_losses/tf_nndistance_so.so: structural_losses/tf_nndistance_g.cu.o structural_losses/tf_nndistance.cpp
	g++ -std=c++11 structural_losses/tf_nndistance.cpp structural_losses/tf_nndistance_g.cu.o -o structural_losses/tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -ltensorflow_framework -L $(TF_LIB) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /usr/local/cuda-9.0/include

structural_losses/tf_nndistance_g.cu.o: structural_losses/tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o structural_losses/tf_nndistance_g.cu.o structural_losses/tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

structural_losses/tf_approxmatch_so.so: structural_losses/tf_approxmatch_g.cu.o structural_losses/tf_approxmatch.cpp
	g++ -std=c++11 structural_losses/tf_approxmatch.cpp structural_losses/tf_approxmatch_g.cu.o -o structural_losses/tf_approxmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /usr/local/cuda-9.0/include

structural_losses/tf_approxmatch_g.cu.o: structural_losses/tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o structural_losses/tf_approxmatch_g.cu.o structural_losses/tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

clean:
	rm structural_losses/tf_approxmatch_so.so
	rm structural_losses/tf_nndistance_so.so
	rm */*.cu.o
