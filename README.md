# Towards Consistent Document-Level Entity Linking: Joint Models for Entity Linking and Coreference Resolution
This repository contains the code, 
dataset, and models for the following paper accepted to ACL 2022 (oral presentation): 
```
@article{zaporojets2021towards,
title = "Towards Consistent Document-level Entity Linking: Joint Models for Entity Linking and Coreference Resolution",
    author = "Zaporojets, Klim  and
      Deleu, Johannes  and
      Jiang, Yiwei  and
      Demeester, Thomas  and
      Develder, Chris",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.88",
    doi = "10.18653/v1/2022.acl-short.88",
    pages = "778--784"
}
```
## GPU Requirements
We have run all the experiments on a single GPU NVIDIA GeForce GTX 1080 (12 Gb of GPU memory).
 
## Creating the Environment
Before proceeding, we recommend creating a separate environment to run the code, and 
then installing the packages in requirements.txt:

```
conda create -n consistent-el python=3.9
conda activate consistent-el
pip install -r requirements.txt
``` 

Install pytorch that corresponds to your cuda version. The default pytorch installation command is: 

```
pip install torch torchvision torchaudio
```



## Datasets
In the present work we use two datasets: 
1. __DWIE__ ([Zaporojets et al., 2021](https://arxiv.org/abs/2009.12626)): this is an entity-centric multi-tasking 
dataset that contains, among others, entity linking and coreference resolution annotations. 
2. __AIDA+__: this is a dataset introduced in the current work and is based on the 
 widely used AIDA ([Hoffart et al., 2011](https://aclanthology.org/D11-1072/)) entity linking dataset. 
 We extend AIDA annotations with:
   1. NIL coreference clusters: we grouped all the mentions that are not linked to any entity in Wikipedia 
   (NIL mentions) in coreference clusters. This resulted in 4,284 NIL coreference clusters which are 
   exploited by our joint coreference and entity linking architecture.  
   2. Consistent cluster-driven entity linking annotations: we observed that 
   some entity linking annotations in AIDA are not complete, with only some of the mentions referring 
   to a specific entity annotated in a document. We extended these annotations to make sure that all the 
   coreferent mentions in the document are linked to the same entity in the Wikipedia Knowledge Base. 
   This increased the number of linked mentions from 27,817 in AIDA to 28,813 in AIDA+ 
   (see Table 1 in [our paper](https://arxiv.org/pdf/2108.13530.pdf)). 

## Instructions to Download the Datasets
To download DWIE and AIDA+ datasets as well as additional 
 files such as entity embeddings and alias tables, the following script has to be executed: 
```
./scripts/download_data.sh
```
After the script is executed, the directory structure 
should look as follows: 
```
├── data
│   ├── aida+
│   └── dwie
├── embeddings
│   ├── entity_embeddings
│   └── token_embeddings
├── experiments
│   ├── aida-global
│   ├── aida-local 
│   ├── aida-standalone-coreference 
│   ├── aida-standalone-linking 
│   ├── dwie-global 
│   ├── dwie-local 
│   ├── dwie-standalone-coreference 
│   └── dwie-standalone-linking 
├── scripts
└── src
```

## Experiments 
The configuration files (```config.json```) of each of the experiments to reproduce
the results of the paper are located inside the ```experiments/``` directory. The 
names of the experiments are self-explanatory and correspond to the architectures
 described in the paper (_Dtandalone_, _Local_ and _Global_). 

## Training
The training script is located in ```src/train.py```, it takes two arguments: 

1. ```--config_file```: the configuration file to run one of the experiments in ```experiments``` directory
 (the names of experiment config files are self explanatory).  
2. ```--output_path```: the directory where the output model, results and tensorboard logs are going to be 
saved.  

For example, to train a global model on DWIE dataset, we can execute: 
```
mkdir experiments/dwie-global/e1/
python -u src/train.py \
--config_file experiments/dwie-global/config.json \
--output_path experiments/dwie-global/e1/ 2>&1 | tee experiments/dwie-global/e1/output_train.log

```

After the training is finished, the resulting directory tree inside ```--output_path```
subdirectory should look as follows: 
```
├── predictions/
├── stored_models/
├── tensorboard_logs/
├── dictionaries/
├── commit_info.json
└── output_train.log
```
Where ```predictions/``` contains the predictions for each of the subsets of the 
dataset used for training. ```stored_models/``` contains the serialized pytorch 
models. ```tensorboard_logs``` are the logs saved during training. Finally, 
```dictionaries``` subdirectory contains the used dictionaries (e.g., the dictionary of the entities)
that map the human-readable entries to internally used ids by the model.
The ```commit_info.json``` and ```output_train.log``` files contain the commit hash 
and the textual logs respectively.  

To obtain the results reported in the paper, 
we trained 5 different models (initialized with random weights)
for each of the studied architectural setups (_Standalone_, _Local_, and _Global_). The script to do this is: 
 ```
 ./scripts/train_all.sh
 ```
Alternatively, the following scripts allows to train only a single model per architecture:
 ```
 ./scripts/train_once.sh
 ```

## Predicting Using Previously Saved Model
As mentioned above (see __Training__), the trained models are saved into ```stored_models```
inside each of the experiment directories. These models can be loaded and used to 
evaluate a given dataset using ```src/evaluate.py``` script. For example:   
```
python -u src/evaluate.py \
--config_file experiments/aida-global/config.json \
--model_path experiments/aida-global/e1/stored_models/last.model \
--output_path experiments/aida-global/e1_evaluate/ 2>&1 \
| tee experiments/aida-global/e1/output_evaluate.log
```
will use the model stored in ```--model_path``` to predict on the subsets listed 
inside ```trainer.evaluate``` of ```--config_file```, and save the predictions
in ```output_path``` subdirectory. 

## Evaluating the Predictions
We evaluate the predictions using the following metrics: 
1. __Entity Linking mention-based (ELm) F1:__ counts a true positive if the mention boundaries and the mention link 
is correctly predicted.  
2. __Entity Linking entity-based (ELh) F1:__ counts a true positive only if both the coreference cluster
(in terms of all its mention spans boundaries) and the entity link are correctly predicted.
3. __Average Coreference Resolution (Coref Avg.) F1:__ we calculate the average-F1 score of com-
monly used MUC ([Vilain et al., 1995](https://doi.org/10.3115/1072399.1072405)), 
B-cubed ([Bagga and Baldwin, 1998](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.5848&rep=rep1&type=pdf)) 
and CEAFe ([Luo, 2005](https://www.aclweb.org/anthology/H05-1004/)).           

### Single Prediction Evaluation
The most basic evaluation setup consists in 
evaluating the predictions made by a particular model for a specific subset using the following command: 
```
PYTHONPATH=src/ python -u src/stats/results/main_linker_results_single.py \
    --predicted_path <<path to the .jsonl file with the predictions>> \
    --ground_truth_path <<directory with ground truth files>>
``` 

Example to get metrics for the predictions on _testa_ subset of AIDA+ inside _aida-global_ experiment: 
```
PYTHONPATH=src/ python -u src/stats/results/main_linker_results_single.py \
    --predicted_path experiments/aida-global/e1/predictions/testa/testa.jsonl \
    --ground_truth_path data/aida+/plain/testa/
``` 

Example to get metrics for the predictions on _test_ subset of DWIE inside _dwie-global_ experiment:
```
PYTHONPATH=src/ python -u src/stats/results/main_linker_results_single.py \
    --predicted_path experiments/dwie-global/e1/predictions/test/test.jsonl \
    --ground_truth_path data/dwie/plain_format/data/annos_with_content/
``` 

### All-Inclusive Evaluation 
Furthermore, we provide ```src/stats/results/main_linker_results_table.py``` 
script to evaluate multiple models at once with different trained models per evaluation setup. 
As a result, a table similar to Table 2 in [our paper](https://arxiv.org/pdf/2108.13530.pdf) is generated. 
The following is an example: 
```
PYTHONPATH=src/ python -u src/stats/results/main_linker_results_table.py \
    --config_file experiments/evaluate_config.json
``` 
Where ```experiments/evaluate_config.json``` is the configuration file containing 
the paths to the experimental runs (for each of the architectural setups) to be evaluated 
(see the provided example). The following are the most important elements in this file: 
1. __datasets:__ a list of the datasets to evaluate on with the corresponding paths to ground truth annotations. 
2. __setups:__ a list of architectural setups to evaluate. In the paper we evaluate three of these setups: 
_Standalone_ (separate for coreference and entity linking tasks), _Local_ (joint coreference and entity linking), 
and _Global_ (joint coreference and entity linking). 
3. __predictions:__ the details on the predictions made using the models from each of the training __runs__. 
4. __runs:__ the paths to the directories where the predictions of each of the trained models for a particular
setup and dataset (e.g., obtained by by running ```./train_all.sh``` script) 
were saved. The reported metric is the average of the calculated metrics for 
the predictions in each of the runs. 

## Contact
If you have questions, please e-mail us at <klim.zaporojets@ugent.be>.

## Acknowledgements
Part of the research has received funding from 
(i) the European Union’s Horizon
2020 research and innovation programme under grant agreement no. 761488 for 
the [CPN project](https://www.projectcpn.eu/), and
(ii) the Flemish Government under the "Onderzoeksprogramma Artificiële Intelligentie (AI) Vlaanderen"
programme.
