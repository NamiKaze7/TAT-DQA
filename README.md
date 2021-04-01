
Tag-tree Model
====================

## Requirements

To create an environment with [Python 3.7 MiniConda](https://docs.conda.io/en/latest/miniconda.html) and activate it.

```bash
conda create -n tag-tree 
conda activate tag-tree
pip install -r requirement.txt
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
```

## TAT-QA Dataset

Please find our TAT-QA dataset under the folder `dataset_raw`.

### JSON Schema
```json
{
  "table": {
    "uid": "3ffd9053-a45d-491c-957a-1b2fa0af0570",
    "table": [
      [
        "",
        "2019",
        "2018",
        "2017"
      ],
      [
        "Fixed Price",
        "$  1,452.4",
        "$  1,146.2",
        "$  1,036.9"
      ],
      [
        "..."
      ]
    ]
  },
  "paragraphs": [
    {
      "uid": "f4ac7069-10a2-47e9-995c-3903293b3d47",
      "order": 1,
      "text": "Sales by Contract Type: Substantially all of our contracts are fixed-price type contracts. Sales included in Other contract types represent cost plus and time and material type contracts."
    },
    {
      "uid": "79e37805-6558-4a8c-b033-32be6bffef48",
      "order": 2,
      "text": "On a fixed-price type contract, ... The table below presents total net sales disaggregated by contract type (in millions)"
    }
  ],
  "questions": [
    {
      "uid": "f4142349-eb72-49eb-9a76-f3ccb1010cbc",
      "order": 1,
      "question": "In which year is the amount of total sales the largest?",
      "answer": [
        "2019"
      ],
      "derivation": "",
      "answer_type": "span",
      "answer_from": "table-text",
      "rel_paragraphs": [
        "2"
      ],
      "req_comparison": true,
      "scale": ""
    },
    {
      "uid": "eb787966-fa02-401f-bfaf-ccabf3828b23",
      "order": 2,
      "question": "What is the change in Other in 2019 from 2018?",
      "answer": -12.6,
      "derivation": "44.1-56.7",
      "answer_type": "arithmetic",
      "answer_from": "table-text",
      "rel_paragraphs": [
        "2"
      ],
      "req_comparison": false,
      "scale": "million"
    }
  ]
}
```

- `table`: the tabular data in a hybrid context.
  - `uid`: the unique id of a table.
  - `table`: a 2d-array of the table content.
  
- `paragraphs`:the textual data in a hybrid context, the associated paragraphs to the table.
  - `uid`: the unique id of a paragraph.
  - `order`: the order of the paragraph in all associated paragraphs, starting from 1.
  - `text`: the content of the paragraph.
  
- `questions`: the generated questions according to the hybrid context.
  - `uid`: the unique id of a question. 
  - `order`: the order of the question in all generated questions, starting from 1.
  - `question`: the question itself.
  - `answer` : the ground-truth answer.
  - `derivation`: the derivation that can be executed to arrive at the ground-truth answer.
  - `answer_type`: the answer type, including `span`, `spans`, `arithmetic` and `counting`.
  - `answer_from`: the source of the answer, including `table`, `table` and `table-text`.
  - `rel_paragraphs`: the paragraphs that are relied to infer the answer if any.
  - `req_comparison`: a flag indicating if `comparison/sorting` is needed to arrive at the answer (`span` or `spans`).
  - `scale`: the scale of the answer, including `None`, `thousand`, `million`, `billion` and `percent`.


## TagOp Model

### Preprocessing dataset

We heuristicly generate the "facts" and "mapping" fields based on raw dataset, which are stored under the folder of `dataset_tagop`.

### Training & Testing TagOp

We use `RoBERTa` as our encoder to develop our TagOp. 
In addition, using the same command, you can run and test the TagOp model based on different encoders (e.g. Bert) by simply changing the argument `--encoder` to different encoder name.

#### Prepare RoBERTa Model 

- Download roberta model.
 
  `mkdir roberta.large && cd roberta.large `
  
  `wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin`

- Download roberta config file.
  
  `wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json`
  - Modify `config.json` from `"output_hidden_states": false` to `"output_hidden_states": true`.
  
  
- Download roberta vocab files.
  
  `wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json`
  
  `wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt`  

#### Prepare the dataset
```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_bert_dataset.py --input_path ./dataset_tagop --output_dir tag_op/data/[ENCODER NAME]  
--encoder [ENCODER NAME] --mode [train/test]
```

#### Train & Evaluation 
```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/train_eval_bert.py --data_dir tag_op/data/[ENCODER NAME]
--save_dir tag_op/model_[ENCODER NAME]_new_latest --batch_size 48 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4 
--weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 50 --eps 1e-6 
--encoder [ENCODER NAME]
```

#### Testing
```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/predict_bert.py --data_dir tag_op/data/[ENCODER NAME] 
--save_dir tag_op/model_[ENCODER NAME] --eval_batch_size 8 --model_path tag_op/model_[ENCODER NAME]_new_latest --mode 2 --encoder [ENCODER NAME]
```

Note: The training process may take around 2 days using a single 32GB v100.

### Effect of different operators in TagOp 

We conduct 2 group of experiments to evaluate the effectiveness of different operators in TagOp. 

By simply changing the argument `--op_mode` from 1-9 for `--ablation_mode 1` or from 1-10 for `--ablation_mode 2`, you can run those ablation studies.

#### Prepare the dataset

   1. train/validation dataset
        ```bash
        PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_roberta_dataset.py --input_path ./dataset_tagop --output_dir tag_op/data/roberta --op_mode [OP_MODE] --ablation_mode [ABLATION_MODE]
        ```

   2. test dataset

        Because we always run the test on the whole test dataset without filtering operators, you will only need to run this command once to generate test data.
        ```bash
        PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_roberta_test_dataset.py --input_path ./dataset_tagop --output_dir tag_op/data/roberta
        ```
#### Training
```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/train_eval_bert.py --data_dir tag_op/data/roberta --save_dir tag_op/model_roberta_new_latest 
--batch_size 48 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4 --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 
--bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 50 --eps 1e-6 --roberta_model dataset_tagop/roberta.large --op_mode [OP_MODE] --ablation_mode [ABLATION_MODE]
```

#### Testing
```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/predict_bert.py --data_dir tag_op/data/roberta --save_dir tag_op/model_roberta 
--eval_batch_size 8 --model_path tag_op/model_roberta_new_latest --mode 2 --roberta_model dataset_tagop/roberta.large --encoder roberta --op_mode [OP_MODE] --ablation_mode [ABLATION_MODE]
```

----------
* * *
Tag-tree Model (2021/3/31)
====================

## Requirements

To create an environment with [Python 3.7 MiniConda](https://docs.conda.io/en/latest/miniconda.html) and activate it.

```bash
conda create -n tag-tree 
conda activate tag-tree
pip install -r requirement.txt
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
```

## TAT-QA Dataset

Please find our TAT-QA dataset under the folder `dataset_raw`.

### JSON Schema
```json
{
  "table": {
    "uid": "3ffd9053-a45d-491c-957a-1b2fa0af0570",
    "table": [
      [
        "",
        "2019",
        "2018",
        "2017"
      ],
      [
        "Fixed Price",
        "$  1,452.4",
        "$  1,146.2",
        "$  1,036.9"
      ],
      [
        "..."
      ]
    ]
  },
  "paragraphs": [
    {
      "uid": "f4ac7069-10a2-47e9-995c-3903293b3d47",
      "order": 1,
      "text": "Sales by Contract Type: Substantially all of our contracts are fixed-price type contracts. Sales included in Other contract types represent cost plus and time and material type contracts."
    },
    {
      "uid": "79e37805-6558-4a8c-b033-32be6bffef48",
      "order": 2,
      "text": "On a fixed-price type contract, ... The table below presents total net sales disaggregated by contract type (in millions)"
    }
  ],
  "questions": [
    {
      "uid": "f4142349-eb72-49eb-9a76-f3ccb1010cbc",
      "order": 1,
      "question": "In which year is the amount of total sales the largest?",
      "answer": [
        "2019"
      ],
      "derivation": "",
      "answer_type": "span",
      "answer_from": "table-text",
      "rel_paragraphs": [
        "2"
      ],
      "req_comparison": true,
      "scale": ""
    },
    {
      "uid": "eb787966-fa02-401f-bfaf-ccabf3828b23",
      "order": 2,
      "question": "What is the change in Other in 2019 from 2018?",
      "answer": -12.6,
      "derivation": "44.1-56.7",
      "answer_type": "arithmetic",
      "answer_from": "table-text",
      "rel_paragraphs": [
        "2"
      ],
      "req_comparison": false,
      "scale": "million"
    }
  ]
}
```

- `table`: the tabular data in a hybrid context.
  - `uid`: the unique id of a table.
  - `table`: a 2d-array of the table content.
  
- `paragraphs`:the textual data in a hybrid context, the associated paragraphs to the table.
  - `uid`: the unique id of a paragraph.
  - `order`: the order of the paragraph in all associated paragraphs, starting from 1.
  - `text`: the content of the paragraph.
  
- `questions`: the generated questions according to the hybrid context.
  - `uid`: the unique id of a question. 
  - `order`: the order of the question in all generated questions, starting from 1.
  - `question`: the question itself.
  - `answer` : the ground-truth answer.
  - `derivation`: the derivation that can be executed to arrive at the ground-truth answer.
  - `answer_type`: the answer type, including `span`, `spans`, `arithmetic` and `counting`.
  - `answer_from`: the source of the answer, including `table`, `table` and `table-text`.
  - `rel_paragraphs`: the paragraphs that are relied to infer the answer if any.
  - `req_comparison`: a flag indicating if `comparison/sorting` is needed to arrive at the answer (`span` or `spans`).
  - `scale`: the scale of the answer, including `None`, `thousand`, `million`, `billion` and `percent`.


## Newmo Model

### Preprocessing dataset

We heuristicly generate the "facts" and "mapping" fields based on raw dataset, which are stored under the folder of `dataset_tagop`.

### Training & Testing Newmo

We use `RoBERTa` as our encoder to develop our Newmo. 
In addition, using the same command, you can run and test the Newmo model based on different encoders (e.g. Bert) by simply changing the argument `--encoder` to different encoder name.

#### Prepare RoBERTa Model 

- Download roberta model.
 
  `mkdir roberta.large && cd roberta.large `
  
  `wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin`

- Download roberta config file.
  
  `wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json`
  - Modify `config.json` from `"output_hidden_states": false` to `"output_hidden_states": true`.
  
  
- Download roberta vocab files.
  
  `wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json`
  
  `wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt`  

#### Prepare the dataset

We provide a new label `span_pos_label` to denote the span's begin/end position, this label is `(-1, -1)` when `answer-type` is not `span`
```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_newmo_dataset.py --input_path ./dataset_tagop --output_dir tag_op/data/[ENCODER NAME]  
--encoder [ENCODER NAME] --mode [train/test]
```

#### Train & Evaluation 
```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/train_newmo.py --data_dir tag_op/data/[ENCODER NAME]
--save_dir tag_op/model_[ENCODER NAME]_new_latest --batch_size 48 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4 
--weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 50 --eps 1e-6 
--encoder [ENCODER NAME]
```

#### Testing
whaiting to write...

