"""
Dataset classes for common uses
Extended from vanilla PANet code by Wang et al.
"""
import random
import torch

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    Base Dataset
    Args:
        base_dir:
            dataset directory
    """
    def __init__(self, base_dir):
        self._base_dir = base_dir
        self.aux_attrib = {}
        self.aux_attrib_args = {}
        self.ids = []  # must be overloaded in subclass

    def add_attrib(self, key, func, func_args):
        """
        Add attribute to the data sample dict

        Args:
            key:
                key in the data sample dict for the new attribute
                e.g. sample['click_map'], sample['depth_map']
            func:
                function to process a data sample and create an attribute (e.g. user clicks)
            func_args:
                extra arguments to pass, expected a dict
        """
        if key in self.aux_attrib:
            raise KeyError("Attribute '{0}' already exists, please use 'set_attrib'.".format(key))
        else:
            self.set_attrib(key, func, func_args)

    def set_attrib(self, key, func, func_args):
        """
        Set attribute in the data sample dict

        Args:
            key:
                key in the data sample dict for the new attribute
                e.g. sample['click_map'], sample['depth_map']
            func:
                function to process a data sample and create an attribute (e.g. user clicks)
            func_args:
                extra arguments to pass, expected a dict
        """
        self.aux_attrib[key] = func
        self.aux_attrib_args[key] = func_args

    def del_attrib(self, key):
        """
        Remove attribute in the data sample dict

        Args:
            key:
                key in the data sample dict
        """
        self.aux_attrib.pop(key)
        self.aux_attrib_args.pop(key)

    def subsets(self, sub_ids, sub_args_lst=None):
        """
        Create subsets by ids

        Args:
            sub_ids:
                a sequence of sequences, each sequence contains data ids for one subset
            sub_args_lst:
                a list of args for some subset-specific auxiliary attribute function
        """

        indices = [[self.ids.index(id_) for id_ in ids] for ids in sub_ids]
        if sub_args_lst is not None:
            subsets = [Subset(dataset=self, indices=index, sub_attrib_args=args)
                       for index, args in zip(indices, sub_args_lst)]
        else:
            subsets = [Subset(dataset=self, indices=index) for index in indices]
        return subsets

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class ReloadPairedDataset(Dataset):
    """
    Make pairs of data from dataset
    Eable only loading part of the entire data in each epoach and then reload to the next part
    Args:
        datasets:
            source datasets, expect a list of Dataset.
            Each dataset indices a certain class. It contains a list of all z-indices of this class for each scan
        n_elements:
            number of elements in a pair
        curr_max_iters:
            number of pairs in an epoch
        pair_based_transforms:
            some transformation performed on a pair basis, expect a list of functions,
            each function takes a pair sample and return a transformed one.
    """
    def __init__(self, datasets, n_elements, curr_max_iters,
                 pair_based_transforms=None):
        super().__init__()
        self.datasets = datasets
        self.n_datasets = len(self.datasets)
        self.n_data = [len(dataset) for dataset in self.datasets]
        self.n_elements = n_elements
        self.curr_max_iters = curr_max_iters
        self.pair_based_transforms = pair_based_transforms
        self.update_index()

    def update_index(self):
        """
        update the order of batches for the next episode
        """

        # update number of elements for each subset
        if hasattr(self, 'indices'):
            n_data_old = self.n_data # DEBUG
            self.n_data = [len(dataset) for dataset in self.datasets]

        if isinstance(self.n_elements, list):
            self.indices = [[(dataset_idx, data_idx) for i, dataset_idx in enumerate(random.sample(range(self.n_datasets), k=len(self.n_elements))) # select which way(s) to use
                                for data_idx in random.sample(range(self.n_data[dataset_idx]), k=self.n_elements[i])] # for each way, which sample to use
                            for i_iter in range(self.curr_max_iters)] # sample <self.curr_max_iters> iterations

        elif self.n_elements > self.n_datasets:
            raise ValueError("When 'same=False', 'n_element' should be no more than n_datasets")
        else:
            self.indices = [[(dataset_idx, random.randrange(self.n_data[dataset_idx]))
                                for dataset_idx in random.sample(range(self.n_datasets),
                                                                k=n_elements)]
                            for i in range(curr_max_iters)]

    def __len__(self):
        return self.curr_max_iters

    def __getitem__(self, idx):
        sample = [self.datasets[dataset_idx][data_idx]
                  for dataset_idx, data_idx in self.indices[idx]]
        if self.pair_based_transforms is not None:
            for transform, args in self.pair_based_transforms:
                sample = transform(sample, **args)
        return sample

class Subset(Dataset):
    """
    Subset of a dataset at specified indices. Used for seperating a dataset by class in our context

    Args:
        dataset:
            The whole Dataset
        indices:
            Indices of samples of the current class in the entire dataset
        sub_attrib_args:
            Subset-specific arguments for attribute functions, expected a dict
    """
    def __init__(self, dataset, indices, sub_attrib_args=None):
        self.dataset = dataset
        self.indices = indices
        self.sub_attrib_args = sub_attrib_args

    def __getitem__(self, idx):
        if self.sub_attrib_args is not None:
            for key in self.sub_attrib_args:
                # Make sure the dataset already has the corresponding attributes
                # Here we only make the arguments subset dependent
                #   (i.e. pass different arguments for each subset)
                self.dataset.aux_attrib_args[key].update(self.sub_attrib_args[key])
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class ValidationDataset(Dataset):
    """
    Dataset for validation

    Args:
        dataset:
            source dataset with a __getitem__ method
        test_classes:
            test classes
        npart: int. number of parts, used for evaluation when assigning support images

    """
    def __init__(self, dataset, test_classes: list, npart: int):
        super().__init__()
        self.dataset = dataset
        self.__curr_cls = None
        self.test_classes = test_classes
        self.dataset.aux_attrib = None 
        self.npart = npart

    def set_curr_cls(self, curr_cls):
        assert curr_cls in self.test_classes
        self.__curr_cls = curr_cls

    def get_curr_cls(self):
        return self.__curr_cls

    def read_dataset(self):
        """
        override original read_dataset to allow reading with z_margin
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def label_strip(self, label):
        """
        mask unrelated labels out
        """
        out = torch.where(label == self.__curr_cls,
                              torch.ones_like(label), torch.zeros_like(label))
        return out

    def __getitem__(self, idx):
        if self.__curr_cls is None:
            raise Exception("Please initialize current class first")

        sample = self.dataset[idx]
        sample["label"] = self.label_strip( sample["label"] )
        sample["label_t"] = sample["label"].unsqueeze(-1).data.numpy()

        labelname = self.dataset.all_label_names[self.__curr_cls]
        z_min = min(self.dataset.tp1_cls_map[labelname][sample['scan_id']])
        z_max = max(self.dataset.tp1_cls_map[labelname][sample['scan_id']])
        sample["z_min"], sample["z_max"] = z_min, z_max
        try:
            part_assign = int((sample["z_id"] - z_min) // ((z_max - z_min) / self.npart))
        except:
            part_assign = 0
            print("###### DATASET: support only have one valid slice ######")
        if part_assign < 0:
            part_assign = 0
        elif part_assign >= self.npart:
            part_assign = self.npart - 1
        sample["part_assign"] = part_assign

        return sample

