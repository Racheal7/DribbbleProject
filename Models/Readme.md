# Existing Model Usage

The zip file of 25 existing models can be downloaded at [GoogleDrive](https://drive.google.com/file/d/16MRjEjt6XH3_C0p1fw2zhaP40CPxXNha/view?usp=sharing).
Prediction_Example.ipynb shows the example usage of the models. 

### Step 1: Download model
Download the zip file from Google Drive. There are 4 main folders: App, Color, Platform, Screen Func, Screen Layout. Each fold contains a list of subfolders named with tags names. Each subfolder has the following files:
* predict.py: the source code for tag prediction.
* test_img: contains 10 test images.
* model: contains the model.

### Step 2: Put the test image under folder test_img
Select the test images and put them in test_img.

### Step 3: Testing
Run the predict.py and the output would look like:
> Shape of testing datas: (5, 256, 256, 3) <br>
> INFO:tensorflow:Restoring parameters from ecommerce/image_model <br>
> Reload the model from ecommerce/image_model <br>
> 4001113.png     => others <br>
> 4002447.png     => others <br>
> 4003516.jpg     => others <br>
> 4003544.jpg     => ecommerce <br>
> 4005139.jpg     => ecommerce <br>

