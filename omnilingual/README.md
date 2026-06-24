# Omnilingual

This directory focuses on the Omnilingual finetuning. Below it will be detailed how the results can be reproduced. Before running code in this section please make sure that the `data/raw-data` and the `data/clean-data` folders exist. If the `clean-data` folder was not created please refer to the whisper part to create it.

## Virtual Environment

Please set up your virtual environment the first time.

```bash
cd omnilingual
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After it is set up you can just activate it at the start.

```bash
cd omnilingual
source .venv/bin/activate
```

## Data

To preprocess the data run the following script from the `omnilingual` directory:

```bash
python scripts/preprocessing.py
```

After you need to generate the language distribution file:

```bash
python scripts/generate-language-distribution.py
```

After you did this your data folder should look the following way:

```
raw-data/
clean-data/
parquet-data/
├── version=0/
│   ├── corpus=RG/                                          # Rumantsch Grischun
│   │   ├── split=test/language=roh_Latn_ruma1247/
│   │   ├── split=train/language=roh_Latn_ruma1247/
│   │   └── split=dev/language=roh_Latn_ruma1247/
│   │       ├── part-0.parquet
│   │       ├── part-1.parquet
│   │       ├── part-2.parquet
│   │       ├── part-3.parquet
│   │       └── part-4.parquet
│   ├── corpus=Sursilvan/                                   # Sursilvan
│   │   └── ... (same structure)
│   ├── corpus=Vallader/                                    # Vallader
│   ├── corpus=Puter/                                       # Puter
│   ├── corpus=Sutsilvan/                                   # Sutsilvan
│   └── corpus=Surmiran/                                    # Surmiran
└── language_distribution_0.tsv    
.gitkeep
```

## Finetuning

After successful preprocessing we get to the finetuning part. You may set the training configuration in
[romansh-ctc-finetune.yaml](omnilingual_asr/workflows/recipes/wav2vec2/asr/configs/romansh-ctc-finetune.yaml). The Model finetuning can be started via:

```bash
python scripts/finetune.py
```

## Evaluation

To evaluate an official omnilingual model on the romansh test set, set the file name in [scripts/evaluate.py](scripts/evaluate.py) and run:

```bash
python scripts/evaluate.py
```

To evaluate your own finetuned checkpoint, set the checkpoint path in [scripts/evaluate-checkpoint.py](scripts/evaluate-checkpoint.py) file and run:

```bash
python scripts/evaluate-checkpoint.py
```

## Notebooks

All the scripts are also available as jupyter notebooks in the `noteboks` directory. Running the notebooks in sequence is also a way of recreating the results.