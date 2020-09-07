"""
for io with nifti data
"""
import os
import sys
import nibabel as nib
import numpy as np
import pdb
import SimpleITK as sitk

def read_nii_bysitk(input_fid, peel_info = False):
    """ read nii to numpy through simpleitk
    Args:
        peelinfo:   peel direction, origin, spacing and metadata out
    """
    img_obj = sitk.ReadImage(input_fid)
    img_np = sitk.GetArrayFromImage(img_obj)
    if peel_info:
        info_obj = {
                "spacing": img_obj.GetSpacing(),
                "origin": img_obj.GetOrigin(),
                "direction": img_obj.GetDirection(),
                "array_size": img_np.shape
                }
        return img_np, info_obj
    else:
        return img_np

def np2itk(img, ref_obj):
    """
    img: numpy array
    ref_obj: reference sitk object for copying information from
    """
    itk_obj = sitk.GetImageFromArray(img)
    itk_obj.SetSpacing( ref_obj.GetSpacing() )
    itk_obj.SetOrigin( ref_obj.GetOrigin()  )
    itk_obj.SetDirection( ref_obj.GetDirection()  )
    return itk_obj

