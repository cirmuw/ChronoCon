

## Example workflow stolen partially  Konstantin

Goal: Build singularity container on msc cluster

1) Get Access token for your git repos which are not public and enter it in  `sample_file.sif` 
2) Change `build_container_on_msc_cluster.job` accordingly send the build job on msc
3) Success! You can now run commands which you defined in your `pyproject.toml` on the msc cluster by submitting the job via 
```bash 
sbatch run_arbitrary_command_with_container.job   YOUR_COMMAND_HERE
```



### Other notes: 

#### "Debugging": 

It is best to debug your code locally and only run the finished version on the cluster by first: 
- building the container
- submitting the training job (or whatever)

However, rebuilding the container constantly can be annoying, and can be circumvented via
```bash 
export SINGULARITYENV_PYTHONPATH=/msc/home/cwatze93/code/segmentation_ls:$PYTHONPATH
singularity exec --nv   /msc/home/cwatze93/container/cseg_utils.sif \
    python /msc/home/cwatze93/code/segmentation_ls/cseg_utils/tools/print_package_info.py
```
 

#### Give the logs file a name with meaning
It is good practice to read in the parameters from a config file. 
E.g. by adding an argparser to your pythons script like `main.py --config $YOUR_CONFIG_YML`. 

I personally like the name of the logs-file to be derived from the used config file. For instance: 

```bash 
./submit_and_add_args.sh  your_job.job  ./path/to/config_xyz.yml
```

where `your_job.job` contains 
```bash 
#SBATCH -o /msc/home/YOUR_USER/logs/%x.o%A.txt
#SBATCH -e /msc/home/YOUR_USER/logs/%x.o%A.txt


...

config_file="$1"

singularity exec --nv /msc/home/cwatze93/container/cseg_utils.sif \
    entry_point_or_path_to_main_py \
    --config  $config_file  

```


This will automatically generate the logs in `/msc/home/YOUR_USER/logs/your_job__-path-to-config_xyz.o1234.txt`. 
The script can be used with an arbitrary number of arguments. 


#### Make dependent jobs
It can be convenient to combine steps 2 & 3.
Perhaps you just made some changes in your code and committed them and now you want to run the new code on the cluster. 
One way to do this is by making the `build-` and the `run-` slurm job dependent on one another. 
I wrote a small script for this. 
Simply run: 

```bash
./make_two_jobs_dependent.sh  build_container_on_msc_cluster.job   run_cmd.job
```

(I could not not figure out how to pass arguments to the second script in this workflow...)


where  `run.job` might look something like this: 

```bash 
#!/bin/sh
#SBATCH -q a100
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150G
#SBATCH -p gpu
#SBATCH -o /msc/home/YOUR_USER/logs/%x.o%A.txt

# Exit on error
set -e 

# let's assume you defined in your package the command cir_package1_predict 
# (by adding the entry point to the pyproject.toml)
# And that it contains an argparser which takes some config file as an input

singularity exec --nv ~/container/cw_utils.sif \
  cir_package1_predict \
  --config  /msc/home/cwatze93/slurm/config/predict_ABC.json 
```
