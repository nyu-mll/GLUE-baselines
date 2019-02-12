#!/bin/bash

scr_prefix='/misc/vlgscratch4/BowmanGroup/awang/'
gpuid=${2:-0}
seed=${3:-111}

# GloVe, no attn, multi
function glove_noattn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-glove-noattn-lr1e-4-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid}
}

# GloVe, attn, multi
function glove_attn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-glove-attn-lr1e-4-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid}
}

# CoVe, no attn, multi
function cove_noattn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-cove-noattn-lr1e-4-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid}
}

# CoVe, attn, multi
function cove_attn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-cove-attn-lr1e-4-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid}
}

# ELMo, no attn, multi
function elmo_noattn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline-elmo -r multitask-elmo-noattn-lr1e-4-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid}
}

# ELMo, attn, multi
function elmo_attn_multi() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline-elmo -r multitask-elmo-attn-lr1e-4-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid}
}

# GloVe, no attn, single
function glove_noattn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-noattn-lr1e-4-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid}
}

# GloVe, attn, single
function glove_attn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-attn-lr1e-4lr1e-4-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid}
}

# CoVe, no attn, single
function cove_noattn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-attn-lr1e-4-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid}
}

# CoVe, attn, single
function cove_attn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-attn-lr1e-4-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid}
}

# ELMo, no attn, single
function elmo_noattn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-noattn-lr1e-4-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid}
}

# ELMo, attn, single
function elmo_attn_single() {
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-attn-lr1e-4-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid}
}

# Evalute
function evaluate_multi() {
    #./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-glove-noattn-lr1e-4-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 13
    #./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-glove-attn-lr1e-4-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 19
    #./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-cove-noattn-lr1e-4-s${seed} -S ${seed} -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 7
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r multitask-cove-attn-lr1e-4-s333 -S 333 -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -E attn -tm -N 9
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline-elmo -r multitask-elmo-noattn-lr1e-4-s222 -S 222 -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 23
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline-elmo -r multitask-elmo-attn-lr1e-4-s222 -S 222 -T all -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 10000 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 18
}

function evaluate_single_mnli() {
    seed=111
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-glove-noattn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 22
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-glove-attn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 12
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-cove-noattn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 7
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-cove-attn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid} -tm -N 7
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline-elmo -r singletask-elmo-noattn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 14
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline-elmo -r singletask-elmo-attn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 16
    seed=222
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-glove-noattn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 20
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-glove-attn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 15
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-cove-noattn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 7
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-cove-attn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid} -tm -N 7
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline-elmo -r singletask-elmo-noattn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 20
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline-elmo -r singletask-elmo-attn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 14
    seed=333
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-glove-noattn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 14
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-glove-attn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 14
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-cove-noattn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 15
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline -r singletask-cove-attn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid} -tm -N 7
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline-elmo -r singletask-elmo-noattn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 15
    ./run_stuff.sh -P ${scr_prefix} -n mnli-baseline-elmo -r singletask-elmo-attn-lr0.0001-s${seed} -S ${seed} -T mnli -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 2612 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 14
}

function evaluate_single_qnli() {
    seed=111
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-noattn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 8
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-attn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 16
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-noattn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 9
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-attn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid} -tm -N 12
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-noattn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 3
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-attn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 17

    seed=222
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-noattn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 10
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-attn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 18
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-noattn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 10
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-attn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid} -tm -N 7
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-noattn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 17
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-attn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 3

    seed=333
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-noattn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 13
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-glove-attn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 18
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-noattn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 7
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline -r singletask-cove-attn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid} -tm -N 14
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-noattn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 7
    ./run_stuff.sh -P ${scr_prefix} -n qnliv2-baseline-elmo -r singletask-elmo-attn-lr0.0001-s${seed} -S ${seed} -T qnliv2 -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 7
}

function evaluate_single_sst() {
    seed=111
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-glove-noattn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-glove-attn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-cove-noattn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-cove-attn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline-elmo -r singletask-elmo-noattn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline-elmo -r singletask-elmo-attn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 
    seed=222
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-glove-noattn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-glove-attn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-cove-noattn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-cove-attn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline-elmo -r singletask-elmo-noattn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline-elmo -r singletask-elmo-attn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 
    seed=333
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-glove-noattn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-glove-attn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -E attn -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-cove-noattn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline -r singletask-cove-attn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -c -E attn -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline-elmo -r singletask-elmo-noattn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -tm -N 
    ./run_stuff.sh -P ${scr_prefix} -n sst-baseline-elmo -r singletask-elmo-attn-lr0.0001-s${seed} -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 1053 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -E attn -I ${gpuid} -tm -N 
}


# Debug
function debug() {
    ./run_stuff.sh -P ${scr_prefix} -n multitask-baseline -r debug-sst -S ${seed} -T sst -C mlp -o adam -l 1e-4 -h 1500 -D 0.2 -L 2 -H 0 -M percent_tr -B 1 -V 723 -y .2 -K 0 -p 5 -W proportional -s max -q -m -b 128 -eg -G -I ${gpuid} -t
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
elif [ $1 == 'debug' ]; then
    debug
elif [ $1 == 'evaluate-multi' ]; then
    evaluate_multi
elif [ $1 == 'evaluate-single-mnli' ]; then
    evaluate_single_mnli
elif [ $1 == 'evaluate-single-qnli' ]; then
    evaluate_single_qnli
elif [ $1 == 'evaluate-single-sst' ]; then
    evaluate_single_sst

fi
