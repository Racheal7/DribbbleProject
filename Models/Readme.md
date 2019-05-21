# Existing Model Usage

Prediction_Example.ipynb shows the example usage of the models. 

### Step 1: Download model
Choose the tag that you want to predict and download the folder (folders are named with tag names). Along with the model, download the predict.py file.

### Step 2: Set paths
Put all test images under the folder "test_image". Then change the model path.
```python
## Test dataset directory.
test_data_dir = "test_image"
## Model path.
model_path = "ecommerce/image_model" 
```

### Step 3: Testing
Run the code and the output would look like:
> Shape of testing datas: (5, 256, 256, 3) <br>
> INFO:tensorflow:Restoring parameters from ecommerce/image_model <br>
> Reload the model from ecommerce/image_model <br>
> 4001113.png     => others <br>
> 4002447.png     => others <br>
> 4003516.jpg     => others <br>
> 4003544.jpg     => ecommerce <br>
> 4005139.jpg     => ecommerce <br>

