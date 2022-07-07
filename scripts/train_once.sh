#!/usr/bin/env bash

########## Training Standalone AIDA+ Entity Linking

# experiment 1: training with random seed initialization
mkdir experiments/aida-standalone-linking/e1/
python -u src/train.py \
--config_file experiments/aida-standalone-linking/config.json \
--output_path experiments/aida-standalone-linking/e1/ 2>&1 | tee experiments/aida-standalone-linking/e1/output_train.log


########## Training Standalone AIDA+ Coreference Resolution

# experiment 1: training with random seed initialization
mkdir experiments/aida-standalone-coreference/e1/
python -u src/train.py \
--config_file experiments/aida-standalone-coreference/config.json \
--output_path experiments/aida-standalone-coreference/e1/ 2>&1 | tee experiments/aida-standalone-coreference/e1/output_train.log

########## Training Local AIDA+ 

# experiment 1: training with random seed initialization
mkdir experiments/aida-local/e1/
python -u src/train.py \
--config_file experiments/aida-local/config.json \
--output_path experiments/aida-local/e1/ 2>&1 | tee experiments/aida-local/e1/output_train.log

########## Training Global AIDA+

# experiment 1: training with random seed initialization
mkdir experiments/aida-global/e1/
python -u src/train.py \
--config_file experiments/aida-global/config.json \
--output_path experiments/aida-global/e1/ 2>&1 | tee experiments/aida-global/e1/output_train.log

########## Training Standalone DWIE Entity Linking

# experiment 1: training with random seed initialization
mkdir experiments/dwie-standalone-linking/e1/
python -u src/train.py \
--config_file experiments/dwie-standalone-linking/config.json \
--output_path experiments/dwie-standalone-linking/e1/ 2>&1 | tee experiments/dwie-standalone-linking/e1/output_train.log

########## Training Standalone DWIE Coreference Resolution

# experiment 1: training with random seed initialization
mkdir experiments/dwie-standalone-coreference/e1/
python -u src/train.py \
--config_file experiments/dwie-standalone-coreference/config.json \
--output_path experiments/dwie-standalone-coreference/e1/ 2>&1 | tee experiments/dwie-standalone-coreference/e1/output_train.log

########## Training Local DWIE

# experiment 1: training with random seed initialization
mkdir experiments/dwie-local/e1/
python -u src/train.py \
--config_file experiments/dwie-local/config.json \
--output_path experiments/dwie-local/e1/ 2>&1 | tee experiments/dwie-local/e1/output_train.log

########## Training Global DWIE

# experiment 1: training with random seed initialization
mkdir experiments/dwie-global/e1/
python -u src/train.py \
--config_file experiments/dwie-global/config.json \
--output_path experiments/dwie-global/e1/ 2>&1 | tee experiments/dwie-global/e1/output_train.log
