#! /bin/bash

# this should point to the directory of this VEP atlas repo
export vep_atlas_dir=$(pwd)
cd $vep_atlas_dir
pwd
# subject name
export SUBJECT=$1

# where should results be stored
export SUBJECTS_DIR=$2

# loop across right and left hemisphere
for hemi in lh rh
do
    # create cortical VEP parcellation
        echo 'create cortical VEP parcellation'
	python -m $vep_atlas_dir/convert_to_vep_parc.py convert_parc 	\
            ${SUBJECTS_DIR}/${SUBJECT}/label/${hemi}.aparc.a2009s.annot \
            ${SUBJECTS_DIR}/${SUBJECT}/surf/${hemi}.pial				   \
            ${SUBJECTS_DIR}/${SUBJECT}/surf/${hemi}.inflated			   \
            ${hemi}												\
            ${vep_atlas_dir}/data/VepAparcColorLut.txt			\
            ${vep_atlas_dir}/data/VepAtlasRules.txt				\
            ${SUBJECTS_DIR}/${SUBJECT}/label/${hemi}.aparc.vep.annot
done

# map cortical labels into the volume
echo 'map cortical labels into the volume'
mri_aparc2aseg --s ${SUBJECT} --annot aparc.vep --base-offset 70000 \
                --o ${SUBJECTS_DIR}/${SUBJECT}/mri/aparc+aseg.vep.mgz


# create volumetric subcortical VEP parcellation
echo 'create volumetric subcortical VEP parcellation'
python -m $vep_atlas_dir/convert_to_vep_parc convert_seg      \
        ${SUBJECTS_DIR}/${SUBJECT}/mri/aparc+aseg.vep.mgz \
        ${vep_atlas_dir}/data/VepFreeSurferColorLut.txt   \
        ${vep_atlas_dir}/data/VepAtlasRules.txt           \
        ${SUBJECTS_DIR}/${SUBJECT}/mri/aparc+aseg.vep.mgz
