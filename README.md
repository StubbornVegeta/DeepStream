**ENGLISH**  |  **[中文版](./README_CN.md)**

# Introduction

DeepStream is a distributed machine learning training framework that is currently under development. It uses MPI to pass parameters and update gradients between multiple machines, and it also allows for training on a single machine. The features currently supported by DeepStream include:

- Synchronous training
- Asynchronous training
- Data parallelism
- Pipeline model parallelism

There are also several features in the development pipeline:

- Tensor model parallelism
- Passing parameters through Gloo
- Disaster recovery

# Dependency

Currently, DeepStream relies on MPI for parameter synchronization, so you need to install OpenMPI. Please note that you should not install both OpenMPI and MPICH at the same time.

## Ubuntu
```bash
sudo apt install openmpi-bin libopenmpi-dev
```

## Archlinux

```bash
sudo pacman -S openmpi
```


# Build && run

## single-node or single-machine
```bash
# ./build_run.sh <node num>
./build_run.sh 1
```

## multi-node
```bash
./build_run.sh 3
```

# Reference

- [https://foundationsofdl.com/2022/02/12/neural-network-from-scratch-part-5-c-deep-learning-framework-implementation/](https://foundationsofdl.com/2022/02/12/neural-network-from-scratch-part-5-c-deep-learning-framework-implementation/)
