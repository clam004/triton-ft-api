# triton-fastertransformer-api

tutorial on how to deploy a scalable autoregressive causal language model transformer as a fastAPI endpoint using nvidia triton server 

the primary value added is that in addition to simplifying and explaining for the beginner machine learning engineer what is happening in the [NVIDIA blog on triton inference server with faster transformer backend](https://developer.nvidia.com/blog/deploying-gpt-j-and-t5-with-fastertransformer-and-triton-inference-server/) we also do a controlled before and after comparison on a realistic RESTful API that you can put into production

## All steps below in one list

```
1. git clone https://github.com/triton-inference-server/fastertransformer_backend.git
2. cd fastertransformer_backend
3. docker build --rm  --build-arg TRITON_VERSION=22.07 -t triton_with_ft:22.07 -f docker/Dockerfile .
4. docker run -it --rm --gpus=all --shm-size=4G  -v $(pwd):/ft_workspace -p 8000:8000 -p 8001:8001 -p 8002:8002 triton_with_ft:22.07 bash

steps 6 to  are from within the bash session started in step 4

6. cd /ft_workspace
7. git clone https://github.com/NVIDIA/FasterTransformer.git
8. cd FasterTransformer
9. mkdir -p build
10. cd build
11. cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
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

`--gpus device=1` limits the GPUs available within the container to the 2nd GPU (cuda:1)

so that even if you have 2 GPUs in your VM, for example:

```
carson@fara-vm2:/home/victorw/fastertransformer_backend/all_models/gpt/fastertransformer$ nvidia-smi
Thu Dec  1 23:38:27 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.85.02    Driver Version: 510.85.02    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000001:00:00.0 Off |                    0 |
| N/A   31C    P0    34W / 250W |   9434MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  Off  | 00000002:00:00.0 Off |                    0 |
| N/A   30C    P0    24W / 250W |      4MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1218      G   /usr/lib/xorg/Xorg                210MiB |
|    0   N/A  N/A      1600      G   /usr/bin/gnome-shell               12MiB |
|    0   N/A  N/A      3351      C   tritonserver                      691MiB |
|    0   N/A  N/A      3397      C   ...onserver/bin/tritonserver     1581MiB |
|    0   N/A  N/A      4450      C   /opt/miniconda/bin/python3       1733MiB |
|    0   N/A  N/A      4451      C   /opt/miniconda/bin/python3       1733MiB |
|    0   N/A  N/A      4452      C   /opt/miniconda/bin/python3       1733MiB |
|    0   N/A  N/A      4453      C   /opt/miniconda/bin/python3       1733MiB |
|    1   N/A  N/A      1218      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
```

within the container, your GPU 0 is the GPU 1 in the VM

```
root@e825b309c6f8:/ft_workspace# nvidia-smi
Thu Dec  1 23:38:04 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.85.02    Driver Version: 510.85.02    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000002:00:00.0 Off |                    0 |
| N/A   30C    P0    24W / 250W |      4MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

```
sudo docker run -it --rm --gpus=all --shm-size=4G  -v $(pwd):/ft_workspace -p 8000:8000 -p 8001:8001 -p 8002:8002 triton_with_ft:22.07 bash
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

```
git clone https://github.com/NVIDIA/FasterTransformer.git

cd FasterTransformer
```

The Cmake file is broken, use vim or some other text editor to change CMakeLists.txt, and have line 193 changed from

`set(PYTHON_PATH "python" CACHE STRING "Python path")` to `set(PYTHON_PATH "python3")`

```
vim CMakeLists.txt
```

```
root@e825b309c6f8:/ft_workspace/models# python3
Python 3.8.10 (default, Jun 22 2022, 20:18:18) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from transformers import GPT2LMHeadModel
>>> model = GPT2LMHeadModel.from_pretrained('gpt2')
Downloading: 100%|██████████████████████████████████████████████████████████████████████| 665/665 [00:00<00:00, 744kB/s]
Downloading: 100%|███████████████████████████████████████████████████████████████████| 548M/548M [00:06<00:00, 81.3MB/s]
>>> model.save_pretrained('./gpt2')
>>> exit()
root@e825b309c6f8:/ft_workspace/models# ls
gpt2
```

```
cd /ft_workspace
python3 ./FasterTransformer/examples/pytorch/gpt/utils/huggingface_gpt_convert.py -o ./all_models/gpt/fastertransformer/1/ -i ./models/gpt2 -i_g 1
```

-i ./models/gpt2 is for where to find the huggingface parameters relative to within `/ft_workspace/`

-i_g 1 flag is used to indicate number of gpus you are using for inference. I'm using 1. if your gpu supports it, use -weight_data_type fp16 as the accuracy loss is minimal and the speedup is significant

```
root@e825b309c6f8:/ft_workspace/all_models/gpt/fastertransformer/1# cd /ft_workspace
root@e825b309c6f8:/ft_workspace# python3 ./FasterTransformer/examples/pytorch/gpt/utils/huggingface_gpt_convert.py -o ./all_models/gpt/fastertransformer/1/ -i ./models/gpt2 -i_g 1
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
Moving 0 files to the new cache system
0it [00:00, ?it/s]

=============== Argument ===============
saved_dir: ./all_models/gpt/fastertransformer/1/
in_file: ./models/gpt2
trained_gpu_num: 1
infer_gpu_num: 1
processes: 4
weight_data_type: fp32
========================================
Some weights of the model checkpoint at ./models/gpt2 were not used when initializing GPT2Model: ['lm_head.weight']
- This IS expected if you are initializing GPT2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing GPT2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Traceback (most recent call last):
  File "./FasterTransformer/examples/pytorch/gpt/utils/huggingface_gpt_convert.py", line 203, in <module>
    split_and_convert(args)
  File "./FasterTransformer/examples/pytorch/gpt/utils/huggingface_gpt_convert.py", line 161, in split_and_convert
    torch.multiprocessing.set_start_method("spawn")
  File "/usr/lib/python3.8/multiprocessing/context.py", line 243, in set_start_method
    raise RuntimeError('context has already been set')
RuntimeError: context has already been set
```

`/ft_workspace/all_models/gpt/fastertransformer/3/2-gpu# vim config.ini `

```
[gpt]
model_name = ./models/gpt2
head_num = 12
size_per_head = 64
inter_size = 3072
max_pos_seq_len = 1024
num_layer = 12
vocab_size = 50257
start_id = 50256
end_id = 50256
weight_data_type = fp32
```

inter_size = Intermediate size of the feed forward network. It is often set to 4 * head_num * size_per_head

`/ft_workspace/all_models/gpt/fastertransformer/config.pbtxt`

example of config.pbtxt for gpt2-medium
```
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

name: "fastertransformer"
backend: "fastertransformer"
default_model_filename: "gpt2-med"
max_batch_size: 1024

model_transaction_policy {
  decoupled: False
}

input [
  {
    name: "input_ids"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  },
  {
    name: "input_lengths"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
  },
  {
    name: "request_output_len"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  },
  {
    name: "runtime_top_k"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "runtime_top_p"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "beam_search_diversity_rate"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "temperature"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "len_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "repetition_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "random_seed"
    data_type: TYPE_UINT64
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "is_return_log_probs"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "beam_width"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "start_id"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "end_id"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "stop_words_list"
    data_type: TYPE_INT32
    dims: [ 2, -1 ]
    optional: true
  },
  {
    name: "bad_words_list"
    data_type: TYPE_INT32
    dims: [ 2, -1 ]
    optional: true
  },
  {
    name: "prompt_learning_task_name_ids"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "request_prompt_embedding"
    data_type: TYPE_FP16
    dims: [ -1, -1 ]
    optional: true
  },
  {
    name: "request_prompt_lengths"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "request_prompt_type"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  }
]
output [
  {
    name: "output_ids"
    data_type: TYPE_UINT32
    dims: [ -1, -1 ]
  },
  {
    name: "sequence_length"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  },
  {
    name: "cum_log_probs"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "output_log_probs"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }
]
instance_group [
  {
    count: 2
    kind : KIND_CPU
  }
]
parameters {
  key: "tensor_para_size"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "pipeline_para_size"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "data_type"
  value: {
    string_value: "fp16"
  }
}
parameters {
  key: "model_type"
  value: {
    string_value: "GPT"
  }
}
parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/ft_workspace/all_models/gpt/fastertransformer/1/1-gpu/"
  }
}
parameters {
  key: "int8_mode"
  value: {
    string_value: "0"
  }
}
parameters {
  key: "enable_custom_all_reduce"
  value: {
    string_value: "0"
  }
}
```


```
parameters {
  key: "head_num"
  value: {
    string_value: "12"
  }
}
parameters {
  key: "size_per_head"
  value: {
    string_value: "64"
  }
}
parameters {
  key: "inter_size"
  value: {
    string_value: "3072"
  }
}
parameters {
  key: "vocab_size"
  value: {
    string_value: "50304"
  }
}
parameters {
  key: "start_id"
  value: {
    string_value: "50256"
  }
}
parameters {
  key: "end_id"
  value: {
    string_value: "50256"
  }
}
parameters {
  key: "decoder_layers"
  value: {
    string_value: "12"
  }
}
parameters {
  key: "model_name"
  value: {
    string_value: "gpt2"
  }
}
parameters {
  key: "model_type"
  value: {
    string_value: "GPT"
  }
}
parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/ft_workspace/all_models/gpt/fastertransformer/1/1-gpu/"
  }
}
```
