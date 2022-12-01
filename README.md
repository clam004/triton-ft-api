# triton-fastertransformer-api

tutorial on how to deploy a scalable autoregressive causal language model transformer as a fastAPI endpoint using nvidia triton server 

the primary value added is that in addition to simplifying and explaining for the beginner machine learning engineer what is happening in the [NVIDIA blog on triton inference server with faster transformer backend](https://developer.nvidia.com/blog/deploying-gpt-j-and-t5-with-fastertransformer-and-triton-inference-server/) we also do a controlled before and after comparison on a realistic RESTful API that you can put into production

## All steps below in one list

```
1. git clone https://github.com/triton-inference-server/fastertransformer_backend.git
2. cd fastertransformer_backend
3. docker build --rm  --build-arg TRITON_VERSION=22.07 -t triton_with_ft:22.07 -f docker/Dockerfile .
4. docker run -it --rm --gpus device=1 --shm-size=4G  -v $(pwd):/ft_workspace -p 2000:8000 -p 2001:8001 -p 2002:8002 triton_with_ft:22.07 bash
5. git clone https://github.com/NVIDIA/FasterTransformer.git
6. cd FasterTransformer
```

```
```

## Detailed explaination of steps

### Clone fastertransformer_backend repository from GitHub

in your terminal within your desired directory. For example, in the same parent directory for this repo but not inside this repo:

terminal input
```
git clone https://github.com/triton-inference-server/fastertransformer_backend.git

cd fastertransformer_backend
```

### Build Docker Image with Triton and FasterTransformer libraries

```
docker build --rm  --build-arg TRITON_VERSION=22.07 -t triton_with_ft:22.07 -f docker/Dockerfile .
```

### Run Docker Container and start an interactive bash session

```
docker run -it --rm --gpus=all --shm-size=4G  -v /home/victorw/fastertransformer_backend:/ft_workspace -p 8000:8001 -p 8990:8002 triton_with_ft:22.07 bash
```

```
docker run -it --rm --gpus=all --shm-size=4G  -v $(pwd):/ft_workspace -p 8989:8000 my_triton_server:v1 bash
```

```
sudo docker run -it --rm --gpus device=1 --shm-size=4G  -v $(pwd):/ft_workspace -p 2000:8000 -p 2001:8001 -p 2002:8002 triton_with_ft:22.07 bash
```

shm = shared memory device

-p <LOCAL_PORT>:<CONTAINER_PORT>

the above command created a volume with -v, now within the bash session enter inot the shared volume /ft_workspace as shown below

```
carson@my-vm:~/path/to/triton/fastertransformer_backend$ sudo docker run -it --rm --gpus device=1 --shm-size=4G  -v $(pwd):/ft_workspace -p 2000:8000 -p 2001:8001 -p 2002:8002 triton_with_ft:22.07 bash

=============================
== Triton Inference Server ==
=============================

NVIDIA Release 22.07 (build 41737377)
Triton Server Version 2.24.0

Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

NOTE: CUDA Forward Compatibility mode ENABLED.
  Using CUDA 11.7 driver version 515.48.08 with kernel driver version 510.85.02.
  See https://docs.nvidia.com/deploy/cuda-compatibility/ for details.

root@a3dfe2f235e4:/workspace# cd /ft_workspace    
```

error
```
root@a3dfe2f235e4:/workspace# cd /ft_workspace                                                                                     
root@a3dfe2f235e4:/ft_workspace# git clone https://github.com/NVIDIA/FasterTransformer.git
Cloning into 'FasterTransformer'...
remote: Enumerating objects: 4038, done.
remote: Counting objects: 100% (690/690), done.
remote: Compressing objects: 100% (153/153), done.
remote: Total 4038 (delta 586), reused 546 (delta 537), pack-reused 3348
Receiving objects: 100% (4038/4038), 18.81 MiB | 18.87 MiB/s, done.
Resolving deltas: 100% (2645/2645), done.
root@a3dfe2f235e4:/ft_workspace# cd FasterTransformer
root@a3dfe2f235e4:/ft_workspace/FasterTransformer# mkdir -p build
root@a3dfe2f235e4:/ft_workspace/FasterTransformer# cd build
root@a3dfe2f235e4:/ft_workspace/FasterTransformer/build# cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
-- The CXX compiler identification is GNU 9.4.0
-- The CUDA compiler identification is NVIDIA 11.7.99
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- Found CUDA: /usr/local/cuda (found suitable version "11.7", minimum required is "10.2") 
CUDA_VERSION 11 is greater or equal than 11, enable -DENABLE_BF16 flag
-- Add DBUILD_MULTI_GPU, requires MPI and NCCL
-- Found MPI_CXX: /opt/hpcx/ompi/lib/libmpi.so (found version "3.1") 
-- Found MPI: TRUE (found version "3.1")  
-- Found NCCL: /usr/include  
-- Determining NCCL version from /usr/include/nccl.h...
-- Looking for NCCL_VERSION_CODE
-- Looking for NCCL_VERSION_CODE - not found
-- Found NCCL (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libnccl.so.2.12.12)
-- Assign GPU architecture (sm=70,75,80,86)
CMAKE_CUDA_FLAGS_RELEASE: -O3 -DNDEBUG -Xcompiler -O3 --use_fast_math
-- COMMON_HEADER_DIRS: /ft_workspace/FasterTransformer;/usr/local/cuda/include
CMake Error at CMakeLists.txt:199 (message):
  PyTorch >= 1.5.0 is needed for TorchScript mode.


-- Configuring incomplete, errors occurred!
See also "/ft_workspace/FasterTransformer/build/CMakeFiles/CMakeOutput.log".
See also "/ft_workspace/FasterTransformer/build/CMakeFiles/CMakeError.log".
```

### 

inside the Docker container interactive session above