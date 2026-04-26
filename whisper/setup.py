from setuptools import setup, find_packages

setup(
    name='whisper_asr',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.45.0',
        'datasets>=2.14.0',
        'evaluate>=0.4.0',
        'librosa',
        'pandas',
        'numpy',
        'tqdm',
        'jiwer',
        'matplotlib',
    ],
)