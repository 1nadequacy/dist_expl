#!/bin/bash

WORKSPACE=/private/home/denisy/workspace/dist_expl

#DOMAIN="ball_in_cup"
#TASK="catch"
DOMAIN=$1
TASK=$2
CDIR=./local/${DOMAIN}_${TASK}/
mkdir -p ${CDIR}

for EXPL_BONUS in 0 1; do
for DIST in 3; do
for SEED in 1 2; do
  #SUBDIR=sac_avg_expbonus_${EXPL_BONUS}_start_${START}_seed_${SEED}
  SUBDIR=sac_off_expbonus_${EXPL_BONUS}_dist_${DIST}_seed_${SEED}
  #SUBDIR=sac_usepot_${USE_POT}_expbonus_${EXPL_BONUS}_goalbonus_${GOAL_BONUS}_seed_${SEED}
  SAVEDIR=${CDIR}/${SUBDIR}
  mkdir -p ${SAVEDIR}
  python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --env_type dmc \
    --num_candidates 1 \
    --state_buffer_size 10000 \
    --eval_freq 5000 \
    --expl_coef ${EXPL_BONUS} \
    --dist_threshold ${DIST} \
    --start_timesteps 1000 \
    --max_timesteps 10000000 \
    --save_dir ${SAVEDIR} \
    --seed ${SEED} 
done
done
done