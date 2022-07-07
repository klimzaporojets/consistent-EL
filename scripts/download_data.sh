#!/usr/bin/env bash

# downloading AIDA+ , embeddings, alias tables, etc. included inside data.zip.
wget https://cloud.ilabt.imec.be/index.php/s/SAKNqFeZ4LBeHrC/download/data.zip
unzip data.zip
rm data.zip

# downloading DWIE; due to licensing, the script dwie_download.py from the DWIE repository has to be used
python -m spacy download en_core_web_sm
mkdir -p data/dwie/plain_format/
git clone https://github.com/klimzaporojets/DWIE.git data/dwie/plain_format
(cd data/dwie/plain_format/ && python -u src/dwie_download.py --tokenize True)

# parsing DWIE into the bert format, some warnings expected
mkdir -p data/dwie/spanbert_format/
python -u src/main_bert_processor_dwie.py --tokenizer_name bert-base-cased \
                                        --input_dir data/dwie/plain_format/data/annos_with_content/ \
                                        --output_dir data/dwie/spanbert_format/ \
                                        --alias_table_path data/dwie/dwie-alias-table/dwie-alias-table.json \
                                        --max_nr_candidates 16 \
                                        --max_span_length 5 \
                                        --max_seg_len 384