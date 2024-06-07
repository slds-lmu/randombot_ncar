#!/bin/bash

#PBS -N submit_experiments
#PBS -A WYOM0174
#PBS -j oe
#PBS -k eod
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1
#PBS -o submit_experiments.log

### Set temp to scratch
export TMPDIR=/glade/derecho/scratch/${USER}/tmp && mkdir -p ${TMPDIR}

source modules

Rscript submit_experiments.R
