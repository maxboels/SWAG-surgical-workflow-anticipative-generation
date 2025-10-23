#!/bin/bash

# select the file name as the first argument. The file name should be in the expts folder.
CONFIG_NAME=$1

# remove the file extension, replace the underscore with a dash, and assign it to JOB_NAME
# also replace / with a dash
JOB_NAME=$(echo $CONFIG_NAME | cut -f 1 -d '.' | tr '_' '-' | tr '/' '-')

# if job name exists, delete it
runai list | grep $JOB_NAME && runai delete job $JOB_NAME
sleep 3

# runai job scheduler on server
runai submit $JOB_NAME \
       --image aicregistry:5000/mboels:supra \
       --run-as-user \
       --large-shm \
       --gpu 1 \
       --cpu 2 \
       --project mboels \
       -v /nfs:/nfs \
       --command \
       -- bash /nfs/home/mboels/projects/SuPRA/run_file.sh $CONFIG_NAME \
       --backoff-limit 0 \
       #-- interactive
       #--run-as-user \
       ## to get the output of the running job, use and change the following line:
       #--port 8080:80 --service-type loadbalancer --port 8080 --service-type ingress
       #--large-shm \


# wait for 3 seconds and list the jobs
sleep 3
watch runai list
