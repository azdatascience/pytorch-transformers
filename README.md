#Supporting code for paper Ablations over Transformer Models for Biomedical Relationship Extraction

This repo is a fork of HuggingFace/Transformers, with some extensions to demonstrate the code used in the paper
Ablations over Transformer Models for Biomedical Relationship Extraction. 


### `run_semeval.py`: relationship classification using R-Bert

R-BERT is a relationship classification head for BERT and RoBERTa, described [here](https://arxiv.org/pdf/1905.08284.pdf").

This example code fine-tunes R-BERT on the semeval 2010 Task 8 dataset:
 
```bash
python ./examples/run_semeval.py \
--data_dir $SEMEVAL_DIR \
--output_dir $RESULTS_DIR \
--model_name_or_path bert-base-uncased \
--do_train \
--do_eval \
--overwrite_output_dir \
--num_train_epochs 8.0 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--learning_rate 2e-5 \
--max_seq_length 128 \
--task_name semeval2010_task8 \
--train_on_other_labels \
--eval_on_other_labels \
--include_directionality

```
The ```$SEMEVAL_DIR``` should point to the extracted archive.

The ```--include_directionality``` flag trains a classifier using all 18 semeval classes. The 
```--train_on_other_labels``` and ```--eval_on_other_labels``` flags also include instances labeled as 'Other' in the 
training and evaluation respectively. Include all of these to be able to use the official evaluation script. 

Note, although an F1 score is calculated in the python code, two additional files are also written out at the checkpoint 
intervals ```{global_step}_semeval_results.tsv``` that may be used with the official Semeval evaluation script 
(supplied in the semeval data archive). The dataset is available under Creative Commons Atrribution 3.0 
Unported Licence (http://creativecommons.org/licenses/by/3.0/) and is available 
[here](http://docs.google.com/leaf?id=0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk&sort=name&layout=list&num=50). 

for example, using BERT:

```bash
./semeval2010_task8_scorer-v1.2.pl $SEMEVAL_DIR/{global_step}_semeval_results.tsv $SEMEVAL_DIR/TEST_FILE_SEMEVAL_SCRIPT_FORMAT.tsv 
```


 
However, the RoBERTa model can also be used with this head:

```bash
python ./examples/run_semeval.py \
--data_dir $SEMEVAL_DIR \
--output_dir $RESULTS_DIR \
--model_name_or_path roberta-large \
--model_type roberta \
--do_train \
--do_eval \
--overwrite_output_dir \
--num_train_epochs 8.0 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--learning_rate 2e-5 \
--max_seq_length 128 \
--task_name semeval2010_task8 \
--train_on_other_labels \
--eval_on_other_labels \
--include_directionality

```

