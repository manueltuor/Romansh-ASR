# Whisper

This directory focuses on the Whisper finetuning. Below it will be detailed how the results can be reproduced. Before running code in this section please make sure that the `raw-data` folder exists and is organised according to the root level [README.md](../README.md).

## Virtual Environment

Please set up your virtual environment the first time.

```bash
cd whisper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After it is set up you can just activate it at the start.

```bash
cd whisper
source .venv/bin/activate
```

## Data

To preprocess the data run the following script from the `whisper` directory:

```bash
python scripts/preprocessing.py
```

After you did this your data folder should look the following way:

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

After successful preprocessing, model finetuning can be started via:

```bash
python scripts/finetune.py
```

**Note:** If you want to mask attention to future speech, set the `STREAMING` constant to True.

## Idiom Classification

After finetuning the model you can also train an idiom classifier on the encoder embeddings of the finetuned Whisper model. For this run:

```bash
python scripts/train-classifier.py
```

## Evaluation

To evaluate a Whisper model you can use the [scripts/evaluate-whisper.py](scripts/evaluate-whisper.py) script, set the `MODEL_PATH` constant to specify which model you want to evaluate.

```bash
python scripts/evaluate-whisper.py
```

To evaluate your idiom classifier run the following script:

```bash
python scripts/evaluate-classifier.py
```

## Notebooks

All the scripts are also available as jupyter notebooks in the `noteboks` directory. Running the notebooks in sequence is also a way of recreating the results.
