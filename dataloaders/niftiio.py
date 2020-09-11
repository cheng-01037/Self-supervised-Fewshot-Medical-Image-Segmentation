"""
Utils for datasets
"""
import numpy as np

import numpy as np
import SimpleITK as sitk


def read_nii_bysitk(input_fid, peel_info = False):
    """ read nii to numpy through simpleitk
        peelinfo: taking direction, origin, spacing and metadata out
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

def convert_to_sitk(input_mat, peeled_info):
    """
    write a numpy array to sitk image object with essential meta-data
    """
    nii_obj = sitk.GetImageFromArray(input_mat)
    if peeled_info:
        nii_obj.SetSpacing(  peeled_info["spacing"] )
        nii_obj.SetOrigin(   peeled_info["origin"] )
        nii_obj.SetDirection(peeled_info["direction"] )
    return nii_obj



