"""
Utils for datasets
"""
import numpy as np

import os
import sys
import nibabel as nib
import numpy as np
import pdb
import SimpleITK as sitk

DATASET_INFO = {
    "CHAOST2": {
            'PSEU_LABEL_NAME': ["BGD", "SUPFG"],
            'REAL_LABEL_NAME': ["BG", "LIVER", "RK", "LK", "SPLEEN"],
            '_SEP': [0, 4, 8, 12, 16, 20],
            'MODALITY': 'MR',
            'LABEL_GROUP': {
                'pa_all': set(range(1, 5)),
                0: set([1, 4]), # upper_abdomen, leaving kidneies as testing classes
                1: set([2, 3]), # lower_abdomen
                },
            },

    "SABS": {
            'PSEU_LABEL_NAME': ["BGD", "SUPFG"],

            'REAL_LABEL_NAME': ["BGD", "SPLEEN", "KID_R", "KID_l", "GALLBLADDER", "ESOPHAGUS", "LIVER", "STOMACH", "AORTA", "IVC",\
              "PS_VEIN", "PANCREAS", "AG_R", "AG_L"],
            '_SEP': [0, 6, 12, 18, 24, 30],
            'MODALITY': 'CT',
            'LABEL_GROUP':{
                'pa_all': set( [1,2,3,6]  ),
                0: set([1,6]  ), # upper_abdomen: spleen + liver as training, kidneis are testing
                1: set( [2,3] ), # lower_abdomen
                    }
            }

}

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

def get_normalize_op(modality, fids):
    """
    As title
    Args:
        modality:   CT or MR
        fids:       fids for the fold
    """

    def get_CT_statistics(scan_fids):
        """
        As CT are quantitative, get mean and std for CT images for image normalizing
        As in reality we might not be able to load all images at a time, we would better detach statistics calculation with actual data loading
        """
        total_val = 0
        n_pix = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_val += in_img.sum()
            n_pix += np.prod(in_img.shape)
            del in_img
        meanval = total_val / n_pix

        total_var = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_var += np.sum((in_img - meanval) ** 2 )
            del in_img
        var_all = total_var / n_pix

        global_std = var_all ** 0.5

        return meanval, global_std

    if modality == 'MR':

        def MR_normalize(x_in):
            return (x_in - x_in.mean()) / x_in.std()

        return MR_normalize #, {'mean': None, 'std': None} # we do not really need the global statistics for MR

    elif modality == 'CT':
        ct_mean, ct_std = get_CT_statistics(fids)
        # debug
        print(f'###### DEBUG_DATASET CT_STATS NORMALIZED MEAN {ct_mean / 255} STD {ct_std / 255} ######')

        def CT_normalize(x_in):
            """
            Normalizing CT images, based on global statistics
            """
            return (x_in - ct_mean) / ct_std

        return CT_normalize #, {'mean': ct_mean, 'std': ct_std}


