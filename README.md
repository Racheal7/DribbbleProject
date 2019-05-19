# DribbbleProject -- Predict Additional Tags for Dribbble UI Design

## Introduction
Graphical User Interface (GUI) is ubiquitous. With the expansion of shared UI design, image tagging greatly support the UI image search. However, there are two problems with tagging-based search that will  severely degrade its efficiency: abbreviations and synonyms within tag vocabulary, and missing tags. For example, for UI designs, some designers would prefer "ui" tag, while others prefer "user interface" or "user-interface". To overcome these two problems, we construct a folksonomy for UI design based on the existing tags, and develop a CNN model for specifically recommend semantic tags to the existing design. Our model achieve a great performance with the average accuracy as 89.1%. And given a query, our method can retrieve much more data than default search.

## How it works
Through an iterative open coding of thousands of existing tags in the Dribbble UI design dataset which we crawled from Dribbble, we construct a vocabulary of UI semantics with high-level categories. Based on the vocabulary, we train a convolutional neural
network that recommends the missing tags of the UI design. Our CNN model comprises 6 convolutional layers, and each one of them is followed by a max pooling layer with a stride and filter size of 2. Then there are three fully connected layers
containing 2048, 1024, 2 neurons respectively with a dropout rate of 0.5. All layers except the last one are activated by ReLU activation function. And the categorical cross entropy loss was minimized by Adaptive Moment Estimation(Adam) with a learning rate of 0.0001.
