#!bin/env bash 
export SUBJECTS_DIR=output/

ls *.nii | parallel --jobs 5 recon-all -i {} -s {.} -all -qcache -cw256
