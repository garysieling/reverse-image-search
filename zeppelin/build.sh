git clone https://github.com/NVIDIA/nvidia-docker.git

#docker build -t zeppelin-gpu -f Dockerfile-gpu .
docker build -t zeppelin-cpu -f Dockerfile .
