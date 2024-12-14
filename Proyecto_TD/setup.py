# setup.py
from setuptools import setup, find_packages

setup(
    name="proyecto_td",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "pandas",
        "scikit-learn",
        "spacy",
        "nltk",
        "transformers",
        "pymongo",
        "kafka-python",
        "pydantic",
        "pyspark",
        "pytest",
        "gunicorn"
    ],
    entry_points={
        "console_scripts": [
            "proyecto-td=main:main",
        ]
    },
    author="Your Name",
    description="Proyecto Final: Tratamiento de Datos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/proyecto-td",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
