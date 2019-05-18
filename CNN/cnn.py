"""
Created on Tues January 1 11:58:40 2019

@author: Ruiqi Wang
@instition: The Australian National University
"""

"""Import Libraries"""
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from autoaugment import ImageNetPolicy
## Deep Learning Visualization Library
from tf_cnnvis import *


## Training dataset directory.
train_data_dir = "train_dataset_directory"
## Test dataset directory.
test_data_dir = "test_dataset_directory"
## Model path.
model_path = "./model/image_model"

## Mode selection.
train = True  ## training mode.
# train = False  ## testing mode.

## With autoaugment (1) transformation or not(0)
flag = 0

"""Supportive Functions"""
def read_train_data(data_dir, flag):
    """
    This function read in the training dataset.
    INPUT: data_dir -- Training dataset directory.
    OUTPUT: fpaths  -- a list of paths of the files in the dataset.
            datas   -- training data.
            labels  -- corresponding labes for the training data.
    """
    datas = []   ## a list of images
    labels = []  ## a list of labels
    fpaths = []  ## a list of image paths
    
    ## AutoAugment policies.
    policy = ImageNetPolicy()
    
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        
        ## Extract label from file name.
        label = int(fname.split("_")[0])
        
        ## Read image and resize it to 256x256.
        image = Image.open(fpath)
        image = image.resize((256,256),Image.BILINEAR) 
        image = image.convert("RGB")
        
        if flag == 0:
            ## No autoaugmentation
            data = np.array(image) / 255.0 ## Normalization
            datas.append(data)
            labels.append(label)
        else:
            ## With AutoAugment Transformation
            img_list = []
            img_list.append(image) ## add the original image
            labels.append(label)
            ## Randomly select 7 policies.
            for _ in range(7):
                ## Apply autoaugment policies to images.
                img_list.append(policy(image))
                labels.append(label)
            for i in img_list:
                data = np.array(i) / 255.0 
                datas.append(data)
        
    datas = np.array(datas)
    labels = np.array(labels)

    print("Shape of training datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels

def read_test_data(data_dir):
    """
    This function read in the testing dataset.
    INPUT: data_dir -- Test dataset directory.
    OUTPUT: fpaths  -- a list of paths of the files in the dataset.
            fnames  -- a list of file names.
            datas   -- training data.
            labels  -- corresponding labes for the training data.
    """
    datas = []
    labels = []
    fpaths = []
    fnames = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        fnames.append(fname)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)

    print("Shape of testing datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, fnames, datas, labels


def compute_accuracy(v_xs, v_ys):
    """
    This function computes the accuracy of the model.
    INPUT: v_xs -- datas.
           v_ys -- corresponding labels.
    OUTPUT: result -- test accuracy.
    """
    
    global predicted_labels
    global test_paths, test_label
    global num_test_examples

    y_pre = sess.run(predicted_labels, feed_dict={datas_placeholder: v_xs, dropout_placeholdr: 1}) ## dropout rate is 1 when testing
    correct_prediction = tf.equal(y_pre, v_ys)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={datas_placeholder: v_xs, labels_placeholder: v_ys, dropout_placeholdr: 1})
     
    return result

## Record the number of completed epoches.
epochs_completed = 0
## Data index in each epoch.
index_in_epoch = 0

def next_batch(batch_size):
    """
    This function splits out batches of data.
    INPUT:
        batch_size -- the number of data in each batch.
    OUTPUT:
        one batch of data and their labels.
            
    """

    global datas
    global labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    ## When all trainig data have been trained once, the dataset is reordered randomly.   
    if index_in_epoch > num_train_examples:
        epochs_completed += 1   ## finished epoch
        
        ## Shuffle the data
        perm = np.arange(num_train_examples)
        np.random.shuffle(perm)
        datas = datas[perm]
        labels = labels[perm]
        
        ## Start next epoch
        start = 0
        index_in_epoch = batch_size
        
        ## Rise an assert when the batch size is bigger than the number of training data.
        assert batch_size <= num_train_examples
        
    end = index_in_epoch
    
    return datas[start:end], labels[start:end]

"""Load datasets"""
fpaths, datas, labels = read_train_data(train_data_dir, flag)
test_paths, test_names, test_data, test_label = read_test_data(test_data_dir)

num_train_examples = datas.shape[0]     ## Number of training examples
num_test_examples = test_data.shape[0]  ## Number of testing examples
num_classes = len(set(labels))          ## Number of classes

"""------------------------Neural Network-----------------------------"""
## Define placeholders for inputs and labels.
datas_placeholder = tf.placeholder(tf.float32, [None, 256, 256, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])
## Define a placeholder for dropout.
dropout_placeholdr = tf.placeholder(tf.float32)

## Add convolutional layers and max pooling layers.
conv0 = tf.layers.conv2d(datas_placeholder, 32, 5, activation=tf.nn.relu)
pool0 = tf.layers.max_pooling2d(conv0, [2,2], [2, 2])
conv1 = tf.layers.conv2d(pool0, 64, 5, activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [2,2], [2, 2])
conv2 = tf.layers.conv2d(pool1, 128, 5, activation=tf.nn.relu) 
pool2 = tf.layers.max_pooling2d(conv2, [2,2], [2, 2])
conv3 = tf.layers.conv2d(pool2, 256, 3, activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(conv3, [2,2], [2, 2])
conv4 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu)
pool4 = tf.layers.max_pooling2d(conv4, [2,2],[2,2])
conv5 = tf.layers.conv2d(pool4, 1024, 3, activation=tf.nn.relu)
pool5 = tf.layers.max_pooling2d(conv5, [2,2],[2,2])

## Flatten layer.
flatten = tf.layers.flatten(pool5)

## Two fully connected layers.
fc_1 = tf.layers.dense(flatten, 2048, activation = tf.nn.relu)
fc_2 = tf.layers.dense(fc_1, 1024, activation = tf.nn.relu)

## Dropout layer.
dropout_fc = tf.nn.dropout(fc_2, dropout_placeholdr)

## Output layer.
logits = tf.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.arg_max(logits, 1)
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)
mean_loss = tf.reduce_mean(losses)
## AdamOptimizer with learning rate of 0.0001.
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(mean_loss)

# Save the model
saver = tf.train.Saver()

## Uncomment the following line if use GPU.
#tf.device('/gpu:0')

"""---------------------Training and Testing---------------------------------"""
with tf.Session() as sess:
    
    if train:
        
        print("Training Mode")
        sess.run(tf.global_variables_initializer())        ## initialize training variables
        
        epoch = 0 
        number_of_steps = int(num_train_examples) * 50    ## train for 50 epoches
        output_step = int(num_train_examples) / 32        ## number of steps each epoch has
        for step in range(number_of_steps):
            x_batch, y_batch = next_batch(32) ## batch size is 32.
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict={datas_placeholder:x_batch, labels_placeholder:y_batch, dropout_placeholdr: 0.5}) ## dropout rate is 0.5.
            
            ## Output test accuracy and mean loss after each epoch.
            if step % output_step == 0:
                epoch += 1 ## new epoch starts
                acc = compute_accuracy(test_data, test_label)  ## compute the testing accuracy
                print("Epoch = {} \tPredict Accuracy = {} \tMean loss ={}".format(epoch, acc, mean_loss_val))
                
        ## Save model.
        saver.save(sess, model_path)
        
    else:
        
        print("Testing Mode")
        
        # Load the parameters from model path.
        saver.restore(sess, model_path)
        print("Reload the model from {}".format(model_path))
        
        ## Classes names.
        label_name_dict = {
            0: "class_name_for_positive_examples",
            1: "class_name_for_negative_examples"
        }
        
        test_feed_dict = {
            datas_placeholder: test_data,
            labels_placeholder: test_label,
            dropout_placeholdr: 1
        }
        
        ## Run the model
        y_pre = sess.run(predicted_labels, feed_dict=test_feed_dict)
        correct_prediction = tf.equal(y_pre, test_label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={datas_placeholder: test_data, labels_placeholder: test_label, dropout_placeholdr: 1}) ## dropout rate is 0 when testing
        
        # Output accracy, true labels and the predicted labels
        for fpath, fname, real_label, predicted_label in zip(test_paths, test_names, test_label, y_pre):
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            print("{}\t{} => {}".format(fname, real_label_name, predicted_label_name))

        print("Test Accuracy is: {}".format(result))

    """Deep Learning Visualization"""
#        ## Input the images you want to visualize.
#        feed_dict = {datas_placeholder:test_data[0:1], labels_placeholder: test_label[0:1], dropout_placeholdr: 1}
#
#        ## Deconvation visualization
#        layers = ["r", "p", "c"]
#        is_success = deconv_visualization(sess_graph_path = sess, value_feed_dict = feed_dict, 
#                                  input_tensor=datas_placeholder, layers=layers, 
#                                  path_logdir=os.path.join("deconv visualization","img00"), 
#                                  path_outdir=os.path.join("Output","img00"))
#
#        ## Activation visualization
#        is_success = activation_visualization(sess_graph_path = None, value_feed_dict = {datas_placeholder : test_data[0:1]}, 
#                                          layers=layers, path_logdir=os.path.join("activation visualization","img00"), 
#                                          path_outdir=os.path.join("Output","img00"))
