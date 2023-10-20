#!/bin/bash
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.out
#SBATCH --time=4:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --tmp=100000
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --open-mode=truncate

trap "echo sigterm recieved, exiting!" SIGTERM

DATASET_DIR="h5_womd_hptr"
run () {
python -u src/run.py \
trainer=womd \
model=scr_womd \
datamodule=h5_womd \
resume=sub_womd \
action=validate \
trainer.limit_val_batches=1.0 \
resume.checkpoint=YOUR_WANDB_RUN_NAME:latest \
loggers.wandb.name="hptr_womd_val" \
loggers.wandb.project="hptr_sub" \
loggers.wandb.entity="YOUR_ENTITY" \
datamodule.data_dir=${TMPDIR}/datasets \
hydra.run.dir='/cluster/scratch/zhejzhan/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
}

# ! For AV2 dataset.
# DATASET_DIR="h5_av2_hptr" 
# trainer=av2 \
# model=scr_av2 \
# datamodule=h5_av2 \
# resume=sub_av2 \

# ! For testing.
# action=test \


source /cluster/project/cvl/zhejzhan/apps/miniconda3/etc/profile.d/conda.sh
conda activate hptr

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`

echo START copying data: `date`
mkdir $TMPDIR/datasets
cp /cluster/scratch/zhejzhan/$DATASET_DIR/validation.h5 $TMPDIR/datasets/
cp /cluster/scratch/zhejzhan/$DATASET_DIR/testing.h5 $TMPDIR/datasets/
echo DONE copying: `date`

type run
echo START: `date`
run &
wait
echo DONE: `date`

mkdir -p ./logs/slurm
mv ./logs/$SLURM_JOB_ID.out ./logs/slurm/$SLURM_JOB_ID.out

echo finished at: `date`
exit 0;
