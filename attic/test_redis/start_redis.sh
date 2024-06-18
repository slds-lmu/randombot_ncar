#!/bin/bash

#PBS -N start_redis
#PBS -A WYOM0174
#PBS -j oe
#PBS -k eod
#PBS -q main
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=128
#PBS -o start_redis.log

ml conda
conda activate randombot

hostname -I

redis-server --protected-mode no



