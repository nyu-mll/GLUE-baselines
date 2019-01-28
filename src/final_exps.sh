#!/bin/bash

scr_prefix='/misc/vlgscratch4/BowmanGroup/awang/'
gpuid=${2:-0}
seed=${3:-111}

# GloVe, no attn, multi
function glove_noattn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-glove-noattn-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid}
}

# GloVe, attn, multi
function glove_attn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-glove-attn-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid}
}

# CoVe, no attn, multi
function cove_noattn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-cove-noattn-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid}
}

# CoVe, attn, multi
function cove_attn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-cove-attn-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid}
}

# ELMo, no attn, multi
function elmo_noattn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline-elmo -r multitask-elmo-noattn-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid}
}

# ELMo, attn, multi
function elmo_attn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline-elmo -r multitask-elmo-attn-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid}
}

# GloVe, no attn, single
function glove_noattn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-noattn-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid}
}

# GloVe, attn, single
function glove_attn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-attn-lr1e-4-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid}
}

# CoVe, no attn, single
function cove_noattn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-attn-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid}
}

# CoVe, attn, single
function cove_attn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-attn-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid}
}

# ELMo, no attn, single
function elmo_noattn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-noattn-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid}
}

# ELMo, attn, single
function elmo_attn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-attn-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid}
}


if [ $1 == 'glove-noattn-multi' ]; then
    glove_noattn_multi
elif [ $1 == 'glove-attn-multi' ]; then
    glove_attn_multi
elif [ $1 == 'cove-noattn-multi' ]; then
    cove_noattn_multi
elif [ $1 == 'cove-attn-multi' ]; then
    cove_attn_multi
elif [ $1 == 'elmo-noattn-multi' ]; then
    elmo_noattn_multi
elif [ $1 == 'elmo-attn-multi' ]; then
    elmo_attn_multi
elif [ $1 == 'glove-noattn-single' ]; then
    glove_noattn_single
elif [ $1 == 'glove-attn-single' ]; then
    glove_attn_single
elif [ $1 == 'cove-noattn-single' ]; then
    cove_noattn_single
elif [ $1 == 'cove-attn-single' ]; then
    cove_attn_single
elif [ $1 == 'elmo-noattn-single' ]; then
    elmo_noattn_single
elif [ $1 == 'elmo-attn-single' ]; then
    elmo_attn_single
fi
