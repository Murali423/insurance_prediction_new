from setuptools import find_packages, setup

from typing import List

REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements():
    pass



setup(
    name = 'insurance',
    version= "0.0.1",
    author = "Murali Diyva Teja",
    author_email= "1murali5teja@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),    
)