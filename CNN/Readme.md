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
Before start training, specify whether apply **autoaugment** to the training dataset or not. If the number of training set is less than 500, then autoaugment is recommended to increase the diversity of the dataset.
```python
## With autoaugment transformation (1) or not(0)
flag = 0
```
You can change the batch size. The default batch size is 32.
```python
x_batch, y_batch = next_batch(32) ## batch size is 32.
```
If train with GPU, uncomment the line below.
```python
#tf.device('/gpu:0') ## Choose which GPU to use.
```
### Step 3: Test the model.

Set the "train" parameter to False.
```python
## Mode selection.
train = False
```
You could choose to output label name for each label.
```python
## Label names.
label_name_dict = {
    0: "label_name_for_positive_examples",
    1: "label_name_for_negative_examples"
}
```
### Step 4: Deep learning visualization.
To visualize the trained model, install [tf_cnnvis](https://github.com/InFoCusp/tf_cnnvis) first. The tf_cnnvis library provides three different ways of visualisation. Details can be seen at: [https://github.com/InFoCusp/tf_cnnvis](https://github.com/InFoCusp/tf_cnnvis). 
```python
        ## Input the images you want to visualize.
        feed_dict = {datas_placeholder:test_data[0:1], labels_placeholder: test_label[0:1], dropout_placeholdr: 1}

        ## Deconvation visualization
        layers = ["r", "p", "c"]
        is_success = deconv_visualization(sess_graph_path = sess, value_feed_dict = feed_dict, 
                                  input_tensor=datas_placeholder, layers=layers, 
                                  path_logdir=os.path.join("deconv visualization","img00"), 
                                  path_outdir=os.path.join("Output","img00"))

        ## Activation visualization
        is_success = activation_visualization(sess_graph_path = None, value_feed_dict = {datas_placeholder : test_data[0:1]}, 
                                          layers=layers, path_logdir=os.path.join("activation visualization","img00"), 
                                          path_outdir=os.path.join("Output","img00"))
```
