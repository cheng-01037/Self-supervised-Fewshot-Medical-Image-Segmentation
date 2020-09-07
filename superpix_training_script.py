"""
Extended from original implementation of PANet by Wang et al.
TODO:
1. Wrap trainer into a seperate class
2. Wrap plotting function into seperate functions
3. Unify manual annotation dataset and superpixel dataset
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from models.grid_proto_fewshot import FewShotSeg

from dataloaders.dev_customized_med import med_fewshot, med_fewshot_val
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
import dataloaders.augutils as myaug
from dataloaders.dataset_utils import CLASS_LABELS

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric

from config_ssl_upload import ex

import tqdm
from tensorboardX import SummaryWriter
import SimpleITK as sitk
from torchvision.utils import make_grid

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])

    model = model.cuda()
    model.train()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        raise NotImplementedError
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]

    ### Transforms for data augmentation
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    te_transforms = None

    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    tr_parent = SuperpixelDataset( # base dataset
        which_dataset = baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split = _config['eval_fold'],
        mode='train',
        min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
        transforms=tr_transforms,
        nsup = _config['task']['n_shots'],
        scan_per_load = _config['scan_per_load'],
        exclude_list = _config["exclude_cls_list"],
        superpix_scale = _config["superpix_scale"],
        fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None
    )

    te_dataset, te_parent = med_fewshot_val(
        dataset_name = baseset_name,
        base_dir=_config['path'][baseset_name]['data_dir'],
        idx_split = _config['eval_fold'],
        scan_per_load = _config['scan_per_load'],
        act_labels=test_labels,
        npart = _config['task']['npart'],
        nsup = _config['task']['n_shots'],
        extern_normalize_func = tr_parent.norm_func

    )

    ### dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    testloader = DataLoader(
        te_dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )

    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

    i_iter = 0 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Set validation nodes ######')

    # here nruns can be used to indicate number of scans
    mar_val_metric_node = Metric(max_label=max_label, n_scans= len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])

    # writers for tensorboard and setting it up
    tbfile_dir = os.path.join(  _run.observers[0].dir, 'tboard_file' ); os.mkdir(tbfile_dir)
    tb_writer = SummaryWriter( tbfile_dir  )

    _log.info('###### Training ######')
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        for _, sample_batched in enumerate(trainloader):
            # Prepare input
            i_iter += 1
            # add writers
            support_images = [[shot.cuda() for shot in way]
                              for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]

            query_images = [query_image.cuda()
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

            optimizer.zero_grad()
            # FIXME: in the model definition, filter out the failure case where pseudolabel falls outside of image or too small to calculate a prototype
            try:
                query_pred, align_loss, debug_vis, assign_mats = model(support_images, support_fg_mask, support_bg_mask, query_images, isval = False, val_wsize = None)
            except:
                print('Faulty batch detected, skip')
                continue

            query_loss = criterion(query_pred, query_labels)
            loss = query_loss + align_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)
            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # write to tensorboard
            tb_writer.add_scalar('tr_query_loss', query_loss, i_iter)
            tb_writer.add_scalar('tr_align_loss', align_loss, i_iter)
            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i_iter)

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:
                # FIXME: obviously a bug from the original PANet script while it does not hurt training process. Fix this
                loss = log_loss['loss'] / (i_iter + 1)
                align_loss = log_loss['align_loss'] / (i_iter + 1)
                print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss},')

                # write to tensorboard
                tb_writer.add_image("tr_sup_img", t2n(sample_batched['support_images'][0][0]), i_iter)
                tb_writer.add_image("tr_sup_fg", (t2n(sample_batched['support_mask'][0][0]['fg_mask'])), i_iter)
                tb_writer.add_image("tr_sup_bg", (t2n(sample_batched['support_mask'][0][0]['bg_mask'])), i_iter)
                tb_writer.add_image("tr_qr_img", to01(t2n(sample_batched['query_images'][0])), i_iter)
                tb_writer.add_image("tr_qr_lb", (t2n(sample_batched['query_labels'][0])), i_iter)
                tb_writer.add_image("tr_qr_pred", (t2n(query_pred[0].argmax(dim=0) )), i_iter)
                tb_writer.add_image("tr_qr_logits_fg", to01(t2n(query_pred[0][1])) , i_iter)
                tb_writer.add_image("tr_qr_logits_bg", to01(t2n(query_pred[0][0])) , i_iter)
                tb_writer.add_image("tr_qr_prop_bg", (t2n(assign_mats[0][0])) , i_iter)

            ##### validation in between #####
            if (i_iter + 1) % _config['validation_interval'] == 0:
                _log.info('###### Starting validation ######')
                model.eval()
                mar_val_metric_node.reset()

                val_loss = []
                with torch.no_grad():
                    save_pred_buffer = {} # indexed by class

                    for curr_lb in test_labels:
                        te_dataset.set_curr_cls(curr_lb)
                        support_batched = te_parent.get_support(curr_class = curr_lb, class_idx = [curr_lb], scan_idx = _config["support_idx"], npart=_config['task']['npart'])

                        # way(1 for now) x part x shot x 3 x H x W] #
                        support_images = [[shot.cuda() for shot in way]
                                          for way in support_batched['support_images']] # way x part x [shot x C x H x W]
                        suffix = 'mask'
                        support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                           for way in support_batched['support_mask']]
                        support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                           for way in support_batched['support_mask']]

                        curr_scan_count = -1 # counting for current scan
                        _lb_buffer = {} # indexed by scan

                        # buffers for visualization in tensorboard
                        qry_img_vis = []
                        qry_gth_vis = []
                        qry_pred_vis = []
                        qry_fg_logit_vis = []
                        qry_bg_logit_vis = []
                        sup_img_vis = []
                        sup_lb_vis = []
                        assign_bg_vis = []

                        last_qpart = 0 # used as indicator for adding result to buffer

                        for sample_batched in testloader:

                            _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                            if _scan_id in te_parent.potential_support_sid: # skip the support scan, don't include that to query
                                continue
                            if sample_batched["is_start"]:
                                ii = 0
                                curr_scan_count += 1
                                _scan_id = sample_batched["scan_id"][0]
                                assert len(sample_batched["scan_id"]) < 2,f'for now only support 1shot 1way but {len(_scan_id)} is given'
                                outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                                if baseset_name == 'C0':
                                    assert outsize[-1] < 256
                                    outsize = (256, 256, outsize[-1]) # data format issue
                                else:
                                    assert outsize[0] < 256
                                    outsize = (256, 256, outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
                                _pred = np.zeros( outsize )
                                _pred.fill(np.nan)

                            q_part = sample_batched["part_assign"] # the chunck of query, for assignment with support
                            query_images = [sample_batched['image'].cuda()]
                            query_labels = torch.cat([ sample_batched['label'].cuda()], dim=0)

                            # check input format here and see if multi-shot is working

                            # [way, [part, [shot x C x H x W]]] ->
                            sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
                            sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                            sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

                            query_pred, _, _, assign_mats = model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )

                            # visualization
                            if (last_qpart != q_part) and (last_qpart <= q_part): # show the first validation scan
                                last_qpart = q_part
                                qry_img_vis.append( sample_batched['image'][0]  )
                                qry_fg_logit_vis.append( query_pred[:, 1, ...].cpu()  )
                                qry_bg_logit_vis.append( query_pred[:, 0, ...].cpu()  )
                                qry_pred_vis.append( query_pred.argmax(dim = 1).cpu() )

                                qry_gth_vis.append( query_labels[0].cpu() )
                                sup_img_vis.append( support_images[0][q_part][0][0].cpu() )
                                sup_lb_vis.append( support_fg_mask[0][q_part][0].cpu() )

                                assign_bg_vis.append(  assign_mats[0][0] )

                            query_pred = np.array(query_pred.argmax(dim=1)[0].cpu())
                            _pred[..., ii] = query_pred.copy()

                            # decide whether to record this slice.
                            assert _config['z_margin'] == 0, "Starting and ending slice of the class is assumed to be known beforehand, see Roy et al. MedIA 2020"

                            if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                                mar_val_metric_node.record(query_pred, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count) # record within z_margin
                            else:
                                pass

                            ii += 1
                            # now check data format
                            if sample_batched["is_end"]:
                                if _config['dataset'] != 'C0':
                                    _lb_buffer[_scan_id] = _pred.transpose(2,0,1) # H, W, Z -> to Z H W
                                else:
                                    lb_buffer[_scan_id] = _pred

                        save_pred_buffer[str(curr_lb)] = _lb_buffer
                        tb_writer.add_image(f'te_qry_img_cls{str(curr_lb)}',        make_grid(qry_img_vis, normalize = True), i_iter)
                        tb_writer.add_image(f'te_qry_fg_logits_cls{str(curr_lb)}',  make_grid(qry_fg_logit_vis, normalize = True), i_iter)
                        tb_writer.add_image(f'te_qry_bg_logits_cls{str(curr_lb)}',  make_grid(qry_bg_logit_vis, normalize = True), i_iter)
                        tb_writer.add_image(f'te_qry_pred_cls{str(curr_lb)}',       make_grid(qry_pred_vis), i_iter)
                        tb_writer.add_image(f'te_qry_gth_cls{str(curr_lb)}',        make_grid(qry_gth_vis), i_iter)
                        tb_writer.add_image(f'te_sup_img_cls{str(curr_lb)}',        make_grid(sup_img_vis, normalize = True), i_iter)
                        tb_writer.add_image(f'te_sup_lb_cls{str(curr_lb)}',         make_grid(sup_lb_vis, normalize = True), i_iter)
                        tb_writer.add_image(f'te_assignmat_bg_cls{str(curr_lb)}',   make_grid(assign_bg_vis , normalize = True), i_iter)

                    if _config["save_interm_predicts"]:
                        for curr_lb, _preds in save_pred_buffer.items():
                            for _scan_id, _pred in _preds.items():
                                _pred *= float(curr_lb)
                                itk_pred = sitk.GetImageFromArray(_pred)
                                itk_pred.SetSpacing(  te_dataset.dataset.info_by_scan[_scan_id]["spacing"] )
                                itk_pred.SetOrigin(   te_dataset.dataset.info_by_scan[_scan_id]["origin"] )
                                itk_pred.SetDirection(te_dataset.dataset.info_by_scan[_scan_id]["direction"] )

                                fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'{_scan_id}_{curr_lb}_iter_{i_iter + 1}.nii.gz')
                                sitk.WriteImage(itk_pred, fid, True)
                                _log.info(f'###### {fid} has been saved ######')

                    del save_pred_buffer

                del sample_batched, support_images, support_bg_mask, query_images, query_labels, query_pred

                # compute dice scores by scan
                m_classDice,_, m_meanDice,_, m_rawDice = mar_val_metric_node.get_mDice(labels=sorted(test_labels), n_scan=None, give_raw = True)

                m_classPrec,_, m_meanPrec,_,  m_classRec,_, m_meanRec,_, m_rawPrec, m_rawRec = mar_val_metric_node.get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw = True)

                mar_val_metric_node.reset() # reset this calculation node

                # write validation result to log file
                _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
                _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
                _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

                _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
                _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
                _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

                _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
                _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
                _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

                _log.info(f'mar_val batches classDice: {m_classDice}')
                _log.info(f'mar_val batches meanDice: {m_meanDice}')

                _log.info(f'mar_val batches classPrec: {m_classPrec}')
                _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

                _log.info(f'mar_val batches classRec: {m_classRec}')
                _log.info(f'mar_val batches meanRec: {m_meanRec}')


                for _lb, _dsc in zip(test_labels, m_classDice.tolist()):
                    tb_writer.add_scalar(f'mar_val_dice_class_{str(_lb)}', _dsc, i_iter)

                tb_writer.add_scalar(f'mar_val_dice_fg_mean_{str(_lb)}', m_classDice.mean(), i_iter)

                for _lb, _prec in zip(test_labels, m_classPrec.tolist()):
                    tb_writer.add_scalar(f'mar_val_precision_class_{str(_lb)}', _prec, i_iter)

                for _lb, _rec in zip(test_labels, m_classRec.tolist()):
                    tb_writer.add_scalar(f'mar_val_recall_class_{str(_lb)}', _rec, i_iter)

                print("============ ============")

                _log.info(f'end of validation at iteration {i_iter}')

                model.train()

            ##### save checkpoints #####
            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    trainloader.dataset.reload_buffer()
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')


            if (i_iter - 2) > _config['n_steps']:
                return 1

    _log.info('###### Saving model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))


