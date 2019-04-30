# coding: utf8

from setuptools import setup, find_packages

setup(
    name='bailian_nlp',
    author="Liu Xiang",
    description="百炼NLP包",
    version='0.1',
    packages=find_packages(
        exclude=[
            "examples",
            "examples_elmo",
            "exps"
        ]
    ),
    package_data={
        'bailian_nlp': [
            'datadrive/models/ernie/pos.bin',
            'datadrive/models/ernie/*.json',
            'datadrive/models/ernie/*.txt'
        ],
    }
)
