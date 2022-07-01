# Vehicle Dent Detection - High Level Design

## Introduction

Our Problem Statement was to detect the vehicle damages with 3 classes taken into consideration:

1. Dent

2. Scratch

3. Damage

## Model Deployment

We have trained the mode using **"Google Colab"**. 

## Methodology

We have collected approximately 1000  images from various sources to train the model. Then we have annotated those images using Darvin V7 Labs online annotation tool. Firstly, we have trained on TFOD, but the accuracy was not acceptable. Then we shifted to Detectron2 and this was our final model. Our final solution is on Detectron2 and testing is done on two videos, one is in natural lighting condition and other one is in artificial lighting condition. We avoided different sample cars for testing to avoid any confusions but our model is robust and can be tested on any damage detection sample. 

## Output:

- Frame Rates:

- Accuracy(daylight):

- Accuracy(artificial light):

- Future Enhancements: 
  
     - Dataset should be increased and if possible, collected and annotated manually. 
  
     - TF2 should also be taken into consideration for better prospects. 
  
     - UNet Model can be considered (Also other various models)
  
     - Separate model for detection and classification can be considered so that one model is not overburdened with all the tasks.
  
     - Advanced methods like object tracking and processing of less number of frames can be implemented.

## Steps For Training using Detectron2 :

**Step 1**: Install pytorch and the dependences.

**Step 2**: Install Detectron 2 and dependencies.

**Step 3**: Mount google drive if you have training data in drive else upload the training data on tp colab.

**Step 4**: import pytorch and Detectron 2

**Step 5**: Register the data set a coco instance for training

**Step 6**: Verify the Dataset before training

**Step 7**: Set the configuration parameter

- Initial model 

- Training epoch

- Initial weights

- Dataset

- Output path

**Step8**: Start the training the Model is saved into the OUTPUT directory specified in the configuration.

## Steps For Inference using Detectron2 :

**Step 1**: Install pytorch and the dependences.

**Step 2**: Install Detectron 2 and dependencies.

**Step 3**: Mount google drive if you have Inference data in drive else upload the inference data on to colab.

**Step 4**: Import pytorch and Detectron 2

**Step 5**: Set the configuration parameter

- Initial model 

- Training epoch

- Trained weights

- Dataset

- Output path

**Step 6**: Start the Inference by specifying input file name the output inference video is saved in the output path.
