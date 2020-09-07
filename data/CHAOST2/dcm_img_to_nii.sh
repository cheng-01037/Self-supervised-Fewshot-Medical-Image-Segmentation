#!bin/bash
# Convert dicom-like images to nii files in 3D
# This is the first step for image pre-processing

# Feed path to the downloaded data here
DATAPATH=./MR

# Feed path to the output folder here
OUTPATH=./MR

for sid in $(ls "$DATAPATH")
do
	dcm2nii -o "$DATAPATH/$sid/T2SPIR" "$DATAPATH/$sid/T2SPIR/DICOM_anon";
	find "$DATAPATH/$sid/T2SPIR" -name "*.nii.gz" -exec mv {} "$OUTPATH/niis/T2SPIR/image_$sid.nii.gz" \;
done;


