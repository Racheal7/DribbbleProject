# Normalize abbreviation and synonyms in tag vocabulary

## Why
There are so many different written styles for tags in the dataset since they are contributed by thousands of designers with very
diverse technical and linguistic background. To help normalize these morphological forms, we adopt a semi-automatic method5 [Chen
et al., 2017](#References) which leverage both the semantic and lexical information.

## Usage
We use a Python package -- [DomainThesaurus](https://pypi.org/project/DomainThesaurus/) to normalize tag vocabulary.
The zip file shows an example of how we do it. Below is the description of important files in the zip file.
* data/tags.txt: This file contains all tag information (each line contains all tags for a design).
* code/DST_Example.ipynb: Source code for generating synonms and abbreviations. Models will be generated and stored under folder "models".
* models: Store the trained models.

## **References**
```
@inproceedings{chen2017unsupervised,
  title={Unsupervised software-specific morphological forms inference from informal discussions},
  author={Chen, Chunyang and Xing, Zhenchang and Wang, Ximing},
  booktitle={Proceedings of the 39th International Conference on Software Engineering},
  pages={450--461},
  year={2017},
  organization={IEEE Press}
}
```
