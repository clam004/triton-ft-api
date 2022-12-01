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

### 

inside the Docker container interactive session above