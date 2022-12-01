# triton-fastertransformer-api
tutorial on how to deploy a scalable autoregressive causal language model transformer using nvidia triton server 

### Clone fastertransformer_backend repo from GitHub

```
git clone https://github.com/triton-inference-server/fastertransformer_backend.git

cd fastertransformer_backend

git checkout -b t5_gptj_blog remotes/origin/dev/t5_gptj_blog
```

### Build docker image

```
docker build --rm  --build-arg TRITON_VERSION=22.03 -t triton_with_ft:22.03 -f docker/Dockerfile . 
```

### Run docker container

shm = shared memory device

-p <LOCAL_PORT>:<CONTAINER_PORT>

```
docker run -it --rm --gpus=all --shm-size=4G  -v $(pwd):/ft_workspace -p 8989:8000 triton_with_ft:22.03 bash
```