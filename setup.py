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
            # 'datadrive/models/chinese_L-12_H-768_A-12/pos.bin',
            'datadrive/models/chinese_L-12_H-768_A-12/*.json',
            'datadrive/models/chinese_L-12_H-768_A-12/*.txt'
        ],
    }
)
