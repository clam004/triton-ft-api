# triton-fastertransformer-api

tutorial on how to deploy a scalable autoregressive causal language model transformer as a fastAPI endpoint using nvidia triton server and fastertransformer backend

the primary value added is that in addition to simplifying and explaining for the beginner machine learning engineer what is happening in the [NVIDIA blog on triton inference server with faster transformer backend](https://developer.nvidia.com/blog/deploying-gpt-j-and-t5-with-fastertransformer-and-triton-inference-server/) we also do a controlled before and after comparison on a realistic RESTful API that you can put into production

## Step By Step Instructions

enter into your terminal within your desired directory:

```
1. git clone https://github.com/triton-inference-server/fastertransformer_backend.git
2. cd fastertransformer_backend
```

you may not need to use sudo

in docker build you can choose your triton version by for example doing `--build-arg TRITON_VERSION=22.05` instead, you can also change `ft_triton_2207:v1` to whatever name you want for the image

in docker build to build the image from scratch when you already have nother similar image, use `--no-cache` 

if you have multiple GPUs, in step 4 and _ you can do `--gpus device=1` instead of `--gpus=all` if you want to place this triton server only on the 2nd GPU instead of distributed across all GPUs

port 8000 is used to send http requests to triton. 8001 is used for GRPC requests, which are apparently faster than http requests. 8002 is used for monitering. I use 8001. to route gRPC to port 2001 on your VM do `-p 2001:8001` you may need to do this just to reroute to an available port.

```
3. sudo docker build --rm -t triton_ft:v1 -f docker/Dockerfile .
4. sudo docker run -it --rm --gpus=all --shm-size=4G  -v /path/to/fastertransformer_backend:/ft_workspace -p 8001:8001 -p 8002:8002 triton_ft:v1 bash
```

the next steps are from within the bash session started in step 4

```
6. cd /ft_workspace
7. git clone https://github.com/NVIDIA/FasterTransformer.git
8. cd FasterTransformer
```

vim CMakeLists.txt and change `set(PYTHON_PATH "python" CACHE STRING "Python path")` to `set(PYTHON_PATH "python3")`

```
9. mkdir -p build
10. cd build
11. cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
12. make -j32
13. cd /ft_workspace
14. mkdir models
15. cd models
16. python3
17. from transformers import GPT2LMHeadModel
18. model = GPT2LMHeadModel.from_pretrained('gpt2')
19. model.save_pretrained('./gpt2')
20. exit()
```

you may have to run the next step again if the first attempt fails, in the example below we create a folder of binaries for each layer in `/ft_workspace/all_models/gpt/fastertransformer/1/1-gpu`

```
21. cd /ft_workspace
22. python3 ./FasterTransformer/examples/pytorch/gpt/utils/huggingface_gpt_convert.py -o ./all_models/gpt/fastertransformer/1/ -i ./models/gpt2 -i_g 1
23. cd /ft_workspace/all_models/gpt/fastertransformer
24. vim config.pbtxt
```

using the information in `/ft_workspace/all_models/gpt/fastertransformer/1/1-gpu/config.ini`, update config.pbtxt 

```
exit
```

you are no longer in the bash session. you could replace `$(pwd)` with the full path `/path/to/fastertransformer_backend/`, or within `/path/to/fastertransformer_backend/` run:

```
24. sudo docker run -it --rm --gpus device=1 --shm-size=4G  -v $(pwd):/ft_workspace -p 2001:8001 -p 2002:8002 triton_ft:v1 /opt/tritonserver/bin/tritonserver --log-warning false --model-repository=/ft_workspace/all_models/gpt/
```

keep this terminal open, do not exit this terminal window, a successful deployment would result in output: 

```
I1203 05:18:03.283727 1 grpc_server.cc:4195] Started GRPCInferenceService at 0.0.0.0:8001
I1203 05:18:03.283981 1 http_server.cc:2857] Started HTTPService at 0.0.0.0:8000
I1203 05:18:03.326952 1 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

in another terminal, if you check your `nvidia-smi` you will see the model has been loaded to your GPU or GPUs. if you exit from the above, the models will be off loaded. This step is meant to remain as so while the trion server is running. using docker compose you can keep this running as long as the VM is on using `restart: "unless-stopped"`

the docker compose equivalent of step 24 is:

```
version: "2.3"
services:

  fastertransformer:
    restart: "unless-stopped"
    image: triton_with_ft:22.07
    runtime: nvidia
    ports:
      - 2001:8001
    shm_size: 4gb
    volumes:
      - ${HOST_FT_MODEL_REPO}:/ft_workspace
    command: /opt/tritonserver/bin/tritonserver --log-warning false --model-repository=/ft_workspace/all_models/gpt/
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            device_ids: ['0']
            capabilities: [gpu]
```


```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
