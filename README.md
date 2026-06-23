# Automatic Speech Recognition for Romansh

This project is part of my bachelors thesis at the department of computer linguistics. The goal is to create an automatic speech recognition system for Romansh the 4th official swiss national language. For this project we finetune ASR models using OpenAI's [Whisper](https://github.com/openai/whisper) and Meta's [Omnilingual](https://github.com/facebookresearch/omnilingual-asr).

## Data

The data is from the Romansh radio and television and can be obtained via the [RTR Linguistic API](https://developer.srgssr.ch/en/apis/rtr-linguistic). It should be saved in the `data/raw-data` directory at root level. The exact pipeline for downloading the data is provided under `common/Data_Loading.ipynb`. Your data folder should look like this after you downloaded the data:

```
raw-data/
├── rm-cc-2021-05-28/             # Rumantsch Grischun
│   ├── train.tsv
│   ├── validated.tsv
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

## Whisper

First you should run the code in the Whisper directory since this also includes the data preprocessing. For further information refer to `whisper/README.md`. To create the virtual environment for Whisper run this:

```bash
cd whisper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Omnilingual

After you ran the data preprocessing from the Whisper part, you can run the Omnilingual part. To create the virtual environment for Omnilingual run this:

```bash
cd omnilingual
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Results

### WER on test set

| Model | Sursilvan | Surmiran | Sutsilvan | Puter | Vallader | RG | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Whisper-medium baseline** | 98.08 | 104.69 | 106.68 | 94.91 | 93.12 | 89.23 | 97.97 |
| **Whisper-medium finetuned** | 14.73 | 11.77 | 15.80 | 15.06 | 14.87 | 4.78 | 12.91 |
| **Omnilingual-ctc-1b baseline** | 66.85 | 70.21 | 76.73 | 63.71 | 63.08 | 42.61 | 64.80 |
| **Omnilingual-ctc-1b finetuned** | 28.72 | 31.83 | 34.27 | 27.13 | 24.66 | 10.41 | 26.97 |

<br>

### CER on test set

| Model | Sursilvan | Surmiran | Sutsilvan | Puter | Vallader | RG | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Whisper-medium baseline** | 54.78 | 56.01 | 67.61 | 52.3 | 51.17 | 50.37 | 55.4 |
| **Whisper-medium finetuned** | 4.57 | 4.04 | 5.77 | 4.61 | 5.92 | 1.57 | 4.39 |
| **Omnilingual-ctc-1b baseline** | 20.64 | 25.42 | 29.82 | 18.46 | 20.06 | 11.28 | 21.42 |
| **Omnilingual-ctc-1b finetuned** | 7.57 | 8.79 | 9.26 | 5.91 | 6.20 | 2.37 | 6.91 |
