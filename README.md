### Legacy info: 


## Install instructions: 


Note for future me on installing on `cirpc` which as old CUDA version: 
Check cuda version with `nvidia-smi`. -> 12.2  -> 12.1

```
conda create -n torch-cu121 python=3.10 -y
conda activate torch-cu121
conda install -y -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1
```

The python packages come with an appropriate `pyproject.toml` file, which specifies the dependencies. 
This allows you to install them simply with

