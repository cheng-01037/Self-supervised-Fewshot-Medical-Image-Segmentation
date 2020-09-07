"""
Metrics for computing evalutation results
Modified from vanilla PANet code by Wang et al.
"""

import numpy as np

class Metric(object):
    """
    Compute evaluation result

    Args:
        max_label:
            max label index in the data (0 denoting background)
        n_scans:
            number of test scans
    """
    def __init__(self, max_label=20, n_scans=None):
        self.labels = list(range(max_label + 1))  # all class labels
        self.n_scans = 1 if n_scans is None else n_scans

        # list of list of array, each array save the TP/FP/FN statistic of a testing sample
        self.tp_lst = [[] for _ in range(self.n_scans)]
        self.fp_lst = [[] for _ in range(self.n_scans)]
        self.fn_lst = [[] for _ in range(self.n_scans)]

    def reset(self):
        """
        Reset accumulated evaluation. 
        """
        # assert self.n_scans == 1, 'Should not reset accumulated result when we are not doing one-time batch-wise validation'
        del self.tp_lst, self.fp_lst, self.fn_lst
        self.tp_lst = [[] for _ in range(self.n_scans)]
        self.fp_lst = [[] for _ in range(self.n_scans)]
        self.fn_lst = [[] for _ in range(self.n_scans)]

    def record(self, pred, target, labels=None, n_scan=None):
        """
        Record the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative

        Args:
            pred:
                predicted mask array, expected shape is H x W
            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
        """
        assert pred.shape == target.shape

        if self.n_scans == 1:
            n_scan = 0

        # array to save the TP/FP/FN statistic for each class (plus BG)
        tp_arr = np.full(len(self.labels), np.nan)
        fp_arr = np.full(len(self.labels), np.nan)
        fn_arr = np.full(len(self.labels), np.nan)

        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels

        for j, label in enumerate(labels):
            # Get the location of the pixels that are predicted as class j
            idx = np.where(np.logical_and(pred == j, target != 255))
            pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # Get the location of the pixels that are class j in ground truth
            idx = np.where(target == j)
            target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

            # this should not work: if target_idx_j:  # if ground-truth contains this class
            # the author is adding posion to the code
            tp_arr[label] = len(set.intersection(pred_idx_j, target_idx_j))
            fp_arr[label] = len(pred_idx_j - target_idx_j)
            fn_arr[label] = len(target_idx_j - pred_idx_j)

        self.tp_lst[n_scan].append(tp_arr)
        self.fp_lst[n_scan].append(fp_arr)
        self.fn_lst[n_scan].append(fn_arr)

    def get_mIoU(self, labels=None, n_scan=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]

            # Compute mean IoU classwisely
            # Average across n_scans, then average over classes
            mIoU_class = np.vstack([tp_sum[_scan] / (tp_sum[_scan] + fp_sum[_scan] + fn_sum[_scan])
                                    for _scan in range(self.n_scans)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_scan]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_scan]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

    def get_mDice(self, labels=None, n_scan=None, give_raw = False):
        """
        Compute mean Dice score (in 3D scan level)

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        # NOTE: unverified
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]

            # Average across n_scans, then average over classes
            mDice_class = np.vstack([ 2 * tp_sum[_scan] / ( 2 * tp_sum[_scan] + fp_sum[_scan] + fn_sum[_scan])
                                    for _scan in range(self.n_scans)])
            mDice = mDice_class.mean(axis=1)
            print(mDice_class)
            if not give_raw:
                return (mDice_class.mean(axis=0), mDice_class.std(axis=0),
                    mDice.mean(axis=0), mDice.std(axis=0))
            else:
                return (mDice_class.mean(axis=0), mDice_class.std(axis=0),
                    mDice.mean(axis=0), mDice.std(axis=0), mDice_class)

        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_scan]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_scan]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mDice_class = 2 * tp_sum / ( 2 * tp_sum + fp_sum + fn_sum)
            mDice = mIoU_class.mean()

            return mDice_class, mDice

    def get_mPrecRecall(self, labels=None, n_scan=None, give_raw = False):
        """
        Compute precision and recall

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        # NOTE: unverified
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]

            # Compute mean IoU classwisely
            # Average across n_scans, then average over classes
            mPrec_class = np.vstack([ tp_sum[_scan] / ( tp_sum[_scan] + fp_sum[_scan] )
                                    for _scan in range(self.n_scans)])

            mRec_class = np.vstack([ tp_sum[_scan] / ( tp_sum[_scan] + fn_sum[_scan] )
                                    for _scan in range(self.n_scans)])

            mPrec = mPrec_class.mean(axis=1)
            mRec  = mRec_class.mean(axis=1)
            if not give_raw:
                return (mPrec_class.mean(axis=0), mPrec_class.std(axis=0), mPrec.mean(axis=0), mPrec.std(axis=0), mRec_class.mean(axis=0), mRec_class.std(axis=0), mRec.mean(axis=0), mRec.std(axis=0))
            else:
                return (mPrec_class.mean(axis=0), mPrec_class.std(axis=0), mPrec.mean(axis=0), mPrec.std(axis=0), mRec_class.mean(axis=0), mRec_class.std(axis=0), mRec.mean(axis=0), mRec.std(axis=0), mPrec_class, mRec_class)


        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_scan]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_scan]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mPrec_class = tp_sum / (tp_sum + fp_sum)
            mPrec = mPrec_class.mean()

            mRec_class = tp_sum / (tp_sum + fn_sum)
            mRec = mRec_class.mean()

            return mPrec_class, mPrec, mRec_class, mRec

    def get_mIoU_binary(self, n_scan=None):
        """
        Compute mean IoU for binary scenario
        (sum all foreground classes as one class)
        """
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[_scan]), axis=0)
                      for _scan in range(self.n_scans)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0)
                      for _scan in range(self.n_scans)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[_scan]), axis=0)
                      for _scan in range(self.n_scans)]

            # Sum over all foreground classes
            tp_sum = [np.c_[tp_sum[_scan][0], np.nansum(tp_sum[_scan][1:])]
                      for _scan in range(self.n_scans)]
            fp_sum = [np.c_[fp_sum[_scan][0], np.nansum(fp_sum[_scan][1:])]
                      for _scan in range(self.n_scans)]
            fn_sum = [np.c_[fn_sum[_scan][0], np.nansum(fn_sum[_scan][1:])]
                      for _scan in range(self.n_scans)]

            # Compute mean IoU classwisely and average across classes
            mIoU_class = np.vstack([tp_sum[_scan] / (tp_sum[_scan] + fp_sum[_scan] + fn_sum[_scan])
                                    for _scan in range(self.n_scans)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_scan]), axis=0)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_scan]), axis=0)

            # Sum over all foreground classes
            tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
            fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
            fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]

            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU
