## Automated Speech Recognition for Romansh

This project is part of my bachelors thesis at the department of computer linguistics. The goal is to create automated speech recognition for romansh the 4th official swiss national language.

### Start

First you need to create and activate the python virtual environment, as well as install the requirements via these commands in the terminal.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Before running code, the best available GPU should be selected by running the below bash script for optimal use of resources.

```bash
source set_gpu.sh
```
