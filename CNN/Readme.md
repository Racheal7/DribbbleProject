# Train a new model

The two files in this folder can be used to train a new model for a new tag label.

### Why use binary classifiers?
Each UI design may own many semantic tags such as “web”, “red”, “news”, “signup” and “form”, and these tags are not exclusive from each other. Therefore, they cannot be regarded as equal. In this project, we train a binary classifier for each tag label. Such binary classifier also benefits the system extensibility for new tags, as we only need to train a new binary classifier for the new tag without altering existing models for the existing tags. 

## Usage

### Step 1: Prepare dataset.

