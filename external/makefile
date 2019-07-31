#TF_LIB= /home/dmri/.local/lib/python3.5/site-packages/tensorflow/
#TF_INC = /home/dmri/.local/lib/python3.5/site-packages/tensorflow/include/
#nvcc = /usr/local/cuda-9.0/bin/nvcc
#CUDA_LIB = /usr/local/cuda-9.0/lib64
#CUDA_INC = /usr/lib/cuda-9.0/include

nvcc = /usr/lib/cuda-10.0/bin/nvcc
CUDA_LIB = /usr/lib/cuda-10.0/lib64
CUDA_INC = /usr/lib/cuda-10.0/include
TF_INC = /home/markpp/.local/lib/python3.6/site-packages/tensorflow/include
TF_LIB = /home/markpp/.local/lib/python3.6/site-packages/tensorflow

all: structural_losses/tf_nndistance_so.so structural_losses/tf_nndistance_g.cu.o structural_losses/tf_approxmatch_so.so structural_losses/tf_approxmatch_g.cu.o
.PHONY : all

#-ltensorflow_framework 
structural_losses/tf_nndistance_so.so: structural_losses/tf_nndistance_g.cu.o structural_losses/tf_nndistance.cpp
	g++ -std=c++11 structural_losses/tf_nndistance.cpp structural_losses/tf_nndistance_g.cu.o -o structural_losses/tf_nndistance_so.so -shared -fPIC -I $(TF_INC) -L $(TF_LIB) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem $(CUDA_INC)

structural_losses/tf_nndistance_g.cu.o: structural_losses/tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o structural_losses/tf_nndistance_g.cu.o structural_losses/tf_nndistance_g.cu -I $(TF_INC) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

structural_losses/tf_approxmatch_so.so: structural_losses/tf_approxmatch_g.cu.o structural_losses/tf_approxmatch.cpp
	g++ -std=c++11 structural_losses/tf_approxmatch.cpp structural_losses/tf_approxmatch_g.cu.o -o structural_losses/tf_approxmatch_so.so -shared -fPIC -I $(TF_INC) -lcudart -L $(CUDA_LIB) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem $(CUDA_INC)

structural_losses/tf_approxmatch_g.cu.o: structural_losses/tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o structural_losses/tf_approxmatch_g.cu.o structural_losses/tf_approxmatch_g.cu -I $(TF_INC) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

clean:
	rm structural_losses/tf_approxmatch_so.so
	rm structural_losses/tf_nndistance_so.so
	rm */*.cu.o
