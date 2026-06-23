# Whisper

This directory focuses on the Whisper finetuning. One way to reproduce the results is given by running the jupyter notebooks in `notebooks` in sequence. Before running code in this section please make sure that the `raw-data` folder exists and is organised according to the root level `README.md`.

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

To preprocess the data you will need to run the `notebooks/0_Data_Preprocessing.ipynb` jupyter notebook or the `scripts/preprocessing.py` script. After you did this your data folder should look the following way:

```
raw-data/
clean-data/
├── rm-cc-2021-05-28/             # Rumantsch Grischun
│   ├── train.tsv
│   ├── validation.tsv
│   ├── test.tsv
│   └── clips/
│       └── *.wav
├── rmsursilv-cc-2021-05-28/      # Sursilvan
│   └── ... (same structure)
├── rmvallader-cc-2021-05-28/     # Vallader
├── rmputer-cc-2021-06-11/        # Puter
├── rmsutsilv-cc-2022-05-18/      # Sutsilvan
└── rmsursiv-cc-2021-12-23/       # Surmiran
.gitkeep
```

## Finetuning

After successful preprocessing, model finetuning can be done via `notebooks/3_Whisper_Finetuning.ipynb` or `scripts/finetune.py`. The model will be saved under `models/` and can be evaluated via `notebooks/4_Whisper_Evaluation.ipynb`.

## Idiom Classification

After finetuning the model you can also train an idiom classifier on the encoder embeddings of the finetuned Whisper model. For this run either `notebooks/5_Idiom_Classification.ipynb` or `scripts/train_classifier.py` then evaluate with `notebooks/6_Classifier_Evaluation.ipynb`.