#!bin/env bash 

export T1=$1
export SUBJECT=$2
export SUBJECTS_DIR=data/freesurfer/subjects/

recon-all -i $T1 -s $SUBJECT -all -qcache -cw256
