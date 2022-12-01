# triton-fastertransformer-api

tutorial on how to deploy a scalable autoregressive causal language model transformer as a fastAPI endpoint using nvidia triton server 

the primary value added is that in addition to simplifying and explaining for the beginner machine learning engineer what is happening in the [NVIDIA blog on triton inference server with faster transformer backend](https://developer.nvidia.com/blog/deploying-gpt-j-and-t5-with-fastertransformer-and-triton-inference-server/) we also do a controlled before and after comparison on a realistic RESTful API

### Clone fastertransformer_backend repo from GitHub

in your terminal within your desired directory. For example, in the same parent directory for this repo but not inside this repo:

terminal input
```
git clone https://github.com/triton-inference-server/fastertransformer_backend.git

cd fastertransformer_backend

git checkout -b t5_gptj_blog remotes/origin/dev/t5_gptj_blog
```

expected output
```
Branch 't5_gptj_blog' set up to track remote branch 'dev/t5_gptj_blog' from 'origin'.
Switched to a new branch 't5_gptj_blog'
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