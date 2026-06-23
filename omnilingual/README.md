# Omnilingual

This directory focuses on the Omnilingual finetuning. One way to reproduce the results is given by running the jupyter notebooks in `notebooks` in sequence. Before running code in this section please make sure that the `raw-data` and the `clean-data` folders exist.

## Virtual Environment

Please set up your virtual environment the first time.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After it is set up you can just activate it at the start.

```bash
source .venv/bin/activate
```

## Data

To preprocess the data you will need to run the `notebooks/1_Data_Preprocessing.ipynb` jupyter notebook or the `scripts/preprocessing.py` script after that you need to run the `scripts/generate_language_distribution.py`script. After you did this your data folder should look the following way:

```
raw-data/
clean-data/
parquet-data/
└── version=0/
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

After successful preprocessing, model finetuning can be done via `notebooks/2_Omnilingual_Finetuning.ipynb` or `scripts/finetune.py`. The model will be saved under `models/` and can be evaluated via `notebooks/3_Omnilingual_Evaluation.ipynb`.