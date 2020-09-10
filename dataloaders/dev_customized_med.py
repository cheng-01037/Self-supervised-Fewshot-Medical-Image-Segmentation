"""
Customized dataset. Extended from vanilla PANet script by Wang et al.
"""

import os
import random
import torch
import numpy as np

from dataloaders.common import ReloadPairedDataset, ValidationDataset
from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset

def attrib_basic(_sample, class_id):
    """
    Add basic attribute
    Args:
        _sample: data sample
        class_id: class label asscociated with the data
            (sometimes indicting from which subset the data are drawn)
    """
    return {'class_id': class_id}

def getMaskOnly(label, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask}

def getMasks(*args, **kwargs):
    raise NotImplementedError

def fewshot_pairing(paired_sample, n_ways, n_shots, cnt_query, coco=False, mask_only = True):
    """
    Postprocess paired sample for fewshot settings
    For now only 1-way is tested but we leave multi-way possible (inherited from original PANet)

    Args:
        paired_sample:
            data sample from a PairedDataset
        n_ways:
            n-way few-shot learning
        n_shots:
            n-shot few-shot learning
        cnt_query:
            number of query images for each class in the support set
        coco:
            MS COCO dataset. This is from the original PANet dataset but lets keep it for further extension
        mask_only:
            only give masks and no scribbles/ instances. Suitable for medical images (for now)
    """
    if not mask_only:
        raise NotImplementedError
    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query]) # seperation for supports and queries

    # support class ids
    class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)] # class ids for each image (support and query)

    # support images
    support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)]
                      for i in range(n_ways)] # fetch support images for each class

    # support image labels
    if coco:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]]
                           for j in range(n_shots)] for i in range(n_ways)]
    else:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'] for j in range(n_shots)]
                          for i in range(n_ways)]

    if not mask_only:
        support_scribbles = [[paired_sample[cumsum_idx[i] + j]['scribble'] for j in range(n_shots)]
                             for i in range(n_ways)]
        support_insts = [[paired_sample[cumsum_idx[i] + j]['inst'] for j in range(n_shots)]
                         for i in range(n_ways)]
    else:
        support_insts = []

    # query images, masks and class indices
    query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways)
                    for j in range(cnt_query[i])]
    if coco:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'][class_ids[i]]
                        for i in range(n_ways) for j in range(cnt_query[i])]
    else:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'] for i in range(n_ways)
                        for j in range(cnt_query[i])]
    query_cls_idx = [sorted([0,] + [class_ids.index(x) + 1
                                    for x in set(np.unique(query_label)) & set(class_ids)])
                     for query_label in query_labels]

    ###### Generate support image masks ######
    if not mask_only:
        support_mask = [[getMasks(support_labels[way][shot], support_scribbles[way][shot],
                                 class_ids[way], class_ids)
                         for shot in range(n_shots)] for way in range(n_ways)]
    else:
        support_mask = [[getMaskOnly(support_labels[way][shot],
                                 class_ids[way], class_ids)
                         for shot in range(n_shots)] for way in range(n_ways)]

    ###### Generate query label (class indices in one episode, i.e. the ground truth)######
    query_labels_tmp = [torch.zeros_like(x) for x in query_labels]
    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[query_labels[i] == 255] = 255
        for j in range(n_ways):
            query_label_tmp[query_labels[i] == class_ids[j]] = j + 1

    ###### Generate query mask for each semantic class (including BG) ######
    # BG class
    query_masks = [[torch.where(query_label == 0,
                                torch.ones_like(query_label),
                                torch.zeros_like(query_label))[None, ...],]
                   for query_label in query_labels]
    # Other classes in query image
    for i, query_label in enumerate(query_labels):
        for idx in query_cls_idx[i][1:]:
            mask = torch.where(query_label == class_ids[idx - 1],
                               torch.ones_like(query_label),
                               torch.zeros_like(query_label))[None, ...]
            query_masks[i].append(mask)


    return {'class_ids': class_ids,
            'support_images': support_images,
            'support_mask': support_mask,
            'support_inst': support_insts, # leave these interfaces
            'support_scribbles': support_scribbles, 

            'query_images': query_images,
            'query_labels': query_labels_tmp,
            'query_masks': query_masks,
            'query_cls_idx': query_cls_idx,
           }


def med_fewshot(dataset_name, base_dir, idx_split, mode, scan_per_load,
        transforms, act_labels, n_ways, n_shots, max_iters_per_load, min_fg = '', n_queries=1, fix_parent_len = None, exclude_list = [], **kwargs):
    """
    Dataset wrapper
    Args:
        dataset_name:
            indicates what dataset to use
        base_dir:
            dataset directory
        mode: 
            which mode to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        idx_split:
            index of split
        scan_per_load:
            number of scans to load into memory as the dataset is large
            use that together with reload_buffer
        transforms:
            transformations to be performed on images/masks
        act_labels:
            active labels involved in training process. Should be a subset of all labels
        n_ways:
            n-way few-shot learning, should be no more than # of object class labels
        n_shots:
            n-shot few-shot learning
        max_iters_per_load:
            number of pairs per load (epoch size)
        n_queries:
            number of query images
        fix_parent_len:
            fixed length of the parent dataset
    """
    med_set = ManualAnnoDataset


    mydataset = med_set(which_dataset = dataset_name, base_dir=base_dir, idx_split = idx_split, mode = mode,\
         scan_per_load = scan_per_load, transforms=transforms, min_fg = min_fg, fix_length = fix_parent_len,\
         exclude_list = exclude_list, **kwargs)

    mydataset.add_attrib('basic', attrib_basic, {})

    # Create sub-datasets and add class_id attribute. Here the class file is internally loaded and reloaded inside
    subsets = mydataset.subsets([{'basic': {'class_id': ii}}
        for ii, _ in enumerate(mydataset.label_name)])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries), minlength=n_ways)
    # Number of queries for each way
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query] # <n_shot> supports + <cnt_quert>[i] queries 
    # Create paired dataset. We do not include background.
    paired_data = ReloadPairedDataset([subsets[ii] for ii in act_labels], n_elements=n_elements, curr_max_iters=max_iters_per_load, 
                                pair_based_transforms=[
                                    (fewshot_pairing, {'n_ways': n_ways, 'n_shots': n_shots,
                                        'cnt_query': cnt_query, 'mask_only': True})])
    return paired_data, mydataset

def update_loader_dset(loader, parent_set):
    """
    Update data loader and the parent dataset behind
    Args:
        loader: actual dataloader
        parent_set: parent dataset which actually stores the data
    """
    parent_set.reload_buffer()
    loader.dataset.update_index()
    print(f'###### Loader and dataset have been updated ######' )

def med_fewshot_val(dataset_name, base_dir, idx_split, scan_per_load, act_labels, npart, fix_length = None, nsup = 1, **kwargs):
    """
    validation set for med images
    Args:
        dataset_name:
            indicates what dataset to use
        base_dir:
            SABS dataset directory
        mode: (original split)
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        idx_split:
            index of split
        scan_per_batch:
            number of scans to load into memory as the dataset is large
            use that together with reload_buffer
        act_labels:
            actual labels involved in training process. Should be a subset of all labels
        npart: number of chunks for splitting a 3d volume
        nsup:  number of support scans, equivalent to nshot
    """
    mydataset = ManualAnnoDataset(which_dataset = dataset_name, base_dir=base_dir, idx_split = idx_split, mode = 'val', scan_per_load = scan_per_load, transforms=None, min_fg = 1, fix_length = fix_length, nsup = nsup, **kwargs)
    mydataset.add_attrib('basic', attrib_basic, {})

    valset = ValidationDataset(mydataset, test_classes = act_labels, npart = npart)

    return valset, mydataset

