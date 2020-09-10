# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=1
export CUDA_VISIBLE_DEVICES=$GPUID1

PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="myexperiments"
DATASET='CHAOST2_Superpix'
NWORKER=4

NSTEP=100100
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
VAL_INTERVAL=25000 # interval for running validation
SEED='1234'


ALL_EV=( 0) # 5-fold cross validation (0, 1, 2, 3, 4)
ALL_SCALE=( "MIDDLE") # config of pseudolabels
LABEL_SETS=0 # 0 for lower-abdomen group, 1 for upper-abdomen group
EXCLU='[2,3]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled.
# use [1,4] for upper-abdomen, or [] for setting 1 by Roy et al.
SUPP_ID='[4]' # using the fifth scan in the validation set as support.

echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do
    PREFIX="SCALE_${DATASET}_grid${PROTO_GRID}_f${LABEL_SETS}_scale_${SUPERPIX_SCALE}_decy${DECAY}_ev${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./exps/${CPT}_${SUPERPIX_SCALE}_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir $LOGDIR
    fi

    python3 superpix_training_script.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    validation_interval=$VAL_INTERVAL \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    min_fg_data=1 seed=$SEED \
    validation_interval=$VAL_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID
    done
done
