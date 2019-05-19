# Predict Additional Tags for Dribbble UI Design

## Introduction
Graphical User Interface (GUI) is ubiquitous. With the expansion of shared UI design, image tagging greatly support the UI image search. However, there are two problems with tagging-based search that will  severely degrade its efficiency: abbreviations and synonyms within tag vocabulary, and missing tags. For example, for UI designs, some designers would prefer "ui" tag, while others prefer "user interface" or "user-interface". To overcome these two problems, we construct a folksonomy for UI design based on the existing tags, and develop a CNN model for specifically recommend semantic tags to the existing design. Our model achieve a great performance with the average accuracy as 89.1%. And given a query, our method can retrieve much more data than default search.

## How it works
We first perform an iterative open coding of thousands of existing tags in the Dribbble UI design dataset which we crawled from [Dribbble](https://dribbble.com/). Then we construct a vocabulary of UI semantics with high-level categories. Based on the vocabulary, we train a convolutional neural network that recommends additional tags for UI designs. Our CNN model comprises 6 convolutional layers, 6 max pooling layers, and three fully connected layers. [AutoAugment](https://arxiv.org/abs/1805.09501v1) is used to increase performance.

### **Requirements**
Python       version >= "2.7"
tensorflow   version >= "1.8.0"
[tf_cnnvis](https://github.com/InFoCusp/tf_cnnvis)    version = "1.0.0"
