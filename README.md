# CAMOLO

Camouflage YOLO - (CAMOLO) trains adversarial patches to confuse the [YOLO](https://pjreddie.com/darknet/yolo/) family of object detection algorithms.  This repository is an extension of [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo), with a number of bespoke enhancements as well as incorporation of data pre-processing and evaluation scripts from [yoltv4](https://github.com/avanetten/yoltv4). Enhanc

## 1. Build Docker

First intall [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then run the following commands to create a docker image and environment:

    # build
    cd /camolo/docker
    docker build -t camolo_image .
    
    # run
    nvidia-docker run -itd -v /local_data:/local_data  -p 9111:9111 -ti --ipc=host --name camolo camolo_image
    docker start camolo
    docker attach camolo
    conda activate solaris
    
    # run jupyter notebooks
    cd /camolo/
    jupyter notebook --ip 0.0.0.0 --port=9111 --no-browser --allow-root &
    # now visit:
    http://server_url:9111
 
 
## 2. Train

Edit patch\_config.py and then run train\_patch.py.

    docker attach camolo
    conda activate solaris
    cd /camolo/camolo
    python train_patch.py visdrone_v1_4cat_obj_only_v2

This will train an adversarial patch that when overlaid on an object will (hopefully) fool YOLO.  For example:

![Alt text](docs/images/patch_yolt2_ave_26x26_alpha0p8_32vals_visdrone_v1_4cat_obj_only_v2_epoch20.jpg?raw=true "")
