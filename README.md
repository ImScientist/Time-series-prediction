# Time series prediction (Rossmann Kaggle competition)



### Setup Docker image 

Build a docker image:
```bash
docker build -f Dockerfile -t rossmann_img:0.1 .
```
Test the image:
```bash
docker run -it --rm --name delete_me \
    -v "$(pwd)":/home/rossmann \
    -v "$(pwd)/../data":/home/data \
    -w /home/rossmann \
    rossmann_img:0.1 /bin/bash

# Inside the image
/opt/conda/envs/rossmann/bin/python train/train_01.py \
        --data_dir ../data/rossmann-store-sales/source \
        --max_pdq 4 1 2 \
        --n_stores 4
```
Docker login:
```
docker login -u "$AML_CONTAINER_REGISTRY_USR" \
             -p "$AML_CONTAINER_REGISTRY_PWD" \
             $AML_CONTAINER_REGISTRY_SERVER
```
Tag the image:
```bash
docker tag rossmann_img:0.1 $AML_CONTAINER_REGISTRY_SERVER/rossmann_img:0.1
```
Add the image to the container registry:
```bash
docker push $AML_CONTAINER_REGISTRY_SERVER/rossmann_img:0.1
```
