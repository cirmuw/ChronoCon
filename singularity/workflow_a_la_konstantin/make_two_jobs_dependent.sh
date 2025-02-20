#!/bin/sh
# Notes: run with 
# ./make_two_jobs_dependent.sh   job_script1.sh  job_script2.sh




set -e

extract_job_id() {
    local job_id_raw="$1"
    local job_id=${job_id_raw##* }
    echo "$job_id"
}


js1=$1
js2=$2

echo "running $js1  and then after sucesss $js2"

jobid="$(sbatch $js1)"
echo $jobid; jobid=$(extract_job_id "$jobid"); echo "extracted the id: $jobid"


jobid="$(sbatch  --dependency=afterok:$jobid   $js2)"

echo $jobid; jobid=$(extract_job_id "$jobid"); echo "extracted the id: $jobid"
