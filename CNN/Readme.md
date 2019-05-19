# Train a new model

The two files in this folder can be used to train a new model for a new tag label.

### Why use binary classifiers?
Each UI design may own many semantic tags such as “web”, “red”, “news”, “signup” and “form”, and these tags are not exclusive from each other. Therefore, they cannot be regarded as equal. In this project, we train a binary classifier for each tag label. Such binary classifier also benefits the system extensibility for new tags, as we only need to train a new binary classifier for the new tag without altering existing models for the existing tags. 

## Usage

### Step 1: Dataset preparing.
Collect a set of images and rename them as "labelnumber_filename" (label number = 0 if belongs to positive data, 1 if negative). For instance, image "400000.png" belongs to positive data, then it should be renamed as "0_400000.png". Split the dataset into training dataset and testing dataset.

```python
train_data_dir = "train_dataset_directory"  ## Change it to the directory of training dataset.
test_data_dir = "test_dataset_directory"    ## Change it to the directory of testing dataset.
model_path = "./model/image_model"          ## Specify the path for model saving.
```
Then select the mode: train or test.
```python
## Mode selection.
train = True  ## If true: training mode, else: test mode.
```
### Step 2: Train the model.
Before start training, specify whether apply autoaugment to the training dataset or not. If the number of training set is less than 500, then autoaugment is recommended to increase the diversity of the dataset.
```python
## With autoaugment transformation (1) or not(0)
flag = 0
```

