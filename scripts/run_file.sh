CONFIG_NAME=$1

indir="/nfs/home/mboels/projects/SuPRA"
cd $indir
python /nfs/home/mboels/projects/SuPRA/launch.py -c expts/$CONFIG_NAME.txt -g