#!/bin/bash

WORKSPACE=/private/home/denisy/workspace/dist_expl

#DOMAIN="ball_in_cup"
#TASK="catch"
DOMAIN=$1
TASK=$2
CDIR=./bench_dmc_rl/${DOMAIN}_${TASK}/
mkdir -p ${CDIR}

for START in 10000; do
for EXPL_BONUS in 0; do
for SEED in 1 2 3 4 5 6 7 8 9 10; do
  #SUBDIR=sac_avg_expbonus_${EXPL_BONUS}_start_${START}_seed_${SEED}
  SUBDIR=sac_expbonus_${EXPL_BONUS}_seed_${SEED}
  SAVEDIR=${CDIR}/${SUBDIR}
  mkdir -p ${SAVEDIR}
  JOBNAME=sac_${DOMAIN}_${TASK}_${SUBDIR}
  SCRIPT=${SAVEDIR}/run.sh
  SLURM=${SAVEDIR}/run.slrm
  extra=""
  echo "#!/bin/sh" > ${SCRIPT}
  echo "#!/bin/sh" > ${SLURM}
  echo "#SBATCH --job-name=${JOBNAME}" >> ${SLURM}
  echo "#SBATCH --output=${SAVEDIR}/stdout" >> ${SLURM}
  echo "#SBATCH --error=${SAVEDIR}/stderr" >> ${SLURM}
  echo "#SBATCH --partition=learnfair" >> ${SLURM}
  #echo "#SBATCH --comment=\"I amm having an ICLR workshop deadline on Friday 03 29\"" >> ${SLURM}
  echo "#SBATCH --nodes=1" >> ${SLURM}
  echo "#SBATCH --time=3100" >> ${SLURM}
  echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
  echo "#SBATCH --signal=USR1" >> ${SLURM}
  echo "#SBATCH --gres=gpu:volta:1" >> ${SLURM}
  echo "#SBATCH --mem=100000" >> ${SLURM}
  echo "#SBATCH -c 1" >> ${SLURM}
  echo "srun sh ${SCRIPT}" >> ${SLURM}
  echo "echo \$SLURM_JOB_ID >> ${SAVEDIR}/id" >> ${SCRIPT}
  echo "{ " >> ${SCRIPT}
  echo "nvidia-smi" >> ${SCRIPT}
  echo "cd ${WORKSPACE}" >> ${SCRIPT}
  echo MUJOCO_GL="egl" LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --dmc \
    --num_candidates 1 \
    --eval_freq 5000 \
    --start_timesteps ${START} \
    --max_timesteps 10000000 \
    --expl_coef ${EXPL_BONUS} \
    --save_dir ${SAVEDIR} \
    --seed ${SEED} >> ${SCRIPT}
  echo "kill -9 \$\$" >> ${SCRIPT}
  echo "} & " >> ${SCRIPT}
  echo "child_pid=\$!" >> ${SCRIPT}
  echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
  echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
  echo "while true; do     sleep 1; done" >> ${SCRIPT}
  sbatch --exclude=`cat ~/exclude.txt` ${SLURM}
done
done
done