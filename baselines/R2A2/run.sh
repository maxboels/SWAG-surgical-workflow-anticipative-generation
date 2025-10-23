#!/bin/bash


# select the file name as the first argument. The file name should be in the expts folder.
MODEL_NAME=$1

# select the second argument as the config file name
CONFIG_FILE=$2

# replace the underscore with a dash and the dot with a dash to create a job name.
JOB_NAME=$(echo $CONFIG_FILE | tr '_' '-' | tr '.' '-')

# if job name exists, delete it
runai list | grep $JOB_NAME && runai delete job $JOB_NAME

# wait for 3 seconds
sleep 3

# add jobname to runai submit
# add the config file path to override the default config file (hydra library).
runai submit $JOB_NAME \
       --image aicregistry:5000/mboels:aavt-torch-2.0 \
       --run-as-user \
       --gpu 1 \
       --project mboels \
       -v /nfs:/nfs \
       -- python /nfs/home/mboels/projects/avt_mat/launch.py -c expts/${MODEL_NAME}/${CONFIG_FILE}.txt -g \
       # --node-type dgx2 \


