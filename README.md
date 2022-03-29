# Norwegian GPT-J-6B Demo App

It works by default in CPU, but a `DEVICE` env var can be passed in for CUDA.

Build:

```bash
docker build  . --tag nb-gpt-j
```

Run:

```bash
docker run --rm -it -p 8080:8080 -e MODEL_NAME=NbAiLab/nb-gpt-j-6b -e HF_TOKEN=<api_token> -v $(pwd)/streamlitcache:/home/streamlitapp/.cache/huggingface nb-gpt-j
```

Register:

```bash
gcloud auth configure-docker
docker tag nb-gpt-j gcr.io/<project>/nb-gpt-j 
docker push gcr.io/<project>/nb-gpt-j  
```
