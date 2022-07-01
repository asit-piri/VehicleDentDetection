<h1 align="center"><img src="https://github.com/asit-piri/asitpiri.github.io/blob/main/img/Logo.png" width="60"/></h1>
<h3 align="center" style="color:MediumSeaGreen"> Greetings. It’s great connecting with you.</h3>
<h2 align="center">I'm Asit Piri,<br> AI Product Manager who is passionate about telling stories with data and designing things.</h2>

### My skill sets:
* Product Leadership 
* Agile Product Management
* Data Science
* Machine Learning
* MLOps: Level 2 Automation
* Heroku cloud platform

### My tool sets:
<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.gnu.org/software/bash/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/gnu_bash/gnu_bash-icon.svg" alt="bash" width="40" height="40"/> </a> <a href="https://getbootstrap.com" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/bootstrap/bootstrap-plain-wordmark.svg" alt="bootstrap" width="40" height="40"/> </a> <a href="https://www.w3schools.com/css/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="40" height="40"/> </a> <a href="https://www.docker.com/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> </a> <a href="https://flask.palletsprojects.com/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/pocoo_flask/pocoo_flask-icon.svg" alt="flask" width="40" height="40"/> </a> <a href="https://git-scm.com/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> <a href="https://heroku.com" target="_blank"> <img src="https://www.vectorlogo.zone/logos/heroku/heroku-icon.svg" alt="heroku" width="40" height="40"/> </a> <a href="https://www.w3.org/html/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a> <a href="https://www.linux.org/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> <a href="https://www.mongodb.com/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mongodb/mongodb-original-wordmark.svg" alt="mongodb" width="40" height="40"/> </a> <a href="https://www.mysql.com/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> </a> <a href="https://opencv.org/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/> </a> <a href="https://postman.com" target="_blank"> <img src="https://www.vectorlogo.zone/logos/getpostman/getpostman-icon.svg" alt="postman" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

### My interests:
* Machine Learning
* Deep Learning
* Computer Vision
* MLOps
* Dockers
* Python for Data Dcience


<!-- <hr>

<hr> -->

# Vehicle Dent Detection - High Level Design

## Product Vision
Expedite the claim process cycle.


## Problem Statement
Vehicle Dent Detection is a vision computing algorithm to accurately identify the severity of the damage from the vehicle image(s), whether it is a dent, scratch, or major damage, using the AI/DL object detection technique.

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

# Vehicle Dent Detection - Low Level Design

## Flow Diagram

<img src='assets/training_flow_diagram.jpg'/>

## Dependencies

| Packages/Repos | version | Reason                                                                                      |
| -------------- | ------- | ------------------------------------------------------------------------------------------- |
| torch          | 1.5     | Required by Detectrono2                                                                     |
| torchvision    | 0.6     | Required for basic utilities to aid Object Detection                                        |
| opencv         | any     | For data visualization and for reading images and videos and drawing mask over video output |
| numpy          | any     | Basic utility for mathematical operations                                                   |
| detectron2     |         | Main object detection implementation framework                                              |
| coco dataset   |         | Pre-trained model used coco dataset for training                                            |
| R_50_FPN_3x    |         | Base model used for object detection                                                        |

## Design

There are multiple steps involved in the process of creating and training the model.

### Step1: Data Collection and Annotation

We initially started with only 400 images of "car" type vehicle and manually annotated the images using 'LabelMe'. Initial model was coded in Tensorflow 1.x so the ''LabelMe' annotated images were useful directly.

Later on more data was acquired from Kaggle's data source which had over 800 more images of dented cars. These images were annotated using an sem-automation tool called Darwin.

Later the combination of both the image dataset was used on the 2nd model, i.e. a total of 1200 annotated images. The images annotated using Darwin didn't follow the CoCo dataset format and so a separate script was used to convert the annotated json files into the specific format before being used for training the model.

**Note: All the data contained in the above mentioned datasets only consists of car images and no other vehicle images were included due to lack of variations in damage form with regards to other vehicles. But we expect that the model trained on car images will also work with very less tinkering on other forms of vehicles as well.**

- **Total Data Sources:** 2
  
     - Web Scrapped (initial 400 images) [[Source(data with annotation)](https://drive.google.com/drive/folders/1UEXMt9gc8wk44DOFE4CyKbvrG9J8X7I9?usp=sharing)]
  
     - Kaggle Datasource (additional 800 images) [[Datasource](https://storage.googleapis.com/bucket-8732/car_damage/preprocessed.zip), [Annotations](https://drive.google.com/file/d/1-OSmUR5Ef66-OMuFdINLepmZMcbgoeNE/view?usp=sharing)]

- **Annotation tools used:** 2
  
     - `LabelMe` manual automation tool
  
     - `Darwin` semi-automation tool which required a python script to convert annotated files to required format 

- **Dataset Annotation Standard Format:** CoCo dataset format

## Step2: Data Preprocessing and Augmentation

Most of the data preprocessing and augmentation is handled by the base model's preprocessing pipeline. We just feed the model with the images and the properly coco standard formatted annotation file in json format.

As per the Tensorflow 1.x and Detectron2 base model, the images are first cropped and then resized so as to meet the base CNN model's input requirement. As mentioned, this step is handled by the model's preprocessing pipeline.

For data augmentation we used only 2 types of augmentation

- vertical flip

- rotation

This was also handled by the base model's input pipeline.

**Note: To reduce the time spent on downloading the dataset into the google colab environment and after that training it, we stored the dataset in Google Drive and mounted the drive into the Google Colab notebook environment so that less time is spent in copying the dataset.**

## Step3: Model Creation

Initially the model was created using Tensorflow(ver: 1.x). But the model didn't perform as per the expectations. After some research we shifted to Detectron2 which uses pytorch as the base framework.

The base CNN model used for classification is `R_50_FPN_3x` and the Object detection model used is `Masked_RCNN`.

Total number of models tried:

- Masked RCNN 
  
     - various base models
  
     - Tensorflow 1.x

- Masked RCNN
  
     - R_50_FPN_3x as the base  model
  
     - Detectron2 (pytorch as backend)

## Step4: Model Training

The recent model (masked rcnn implemented using detectron) was trained using the default hyperparameters on the training data using 80% of the total data and 20% as the validation data. It was trained on 60,000 epochs with 512 images as the batch size. The learning rate was initialized with 0.005.

Any number of epochs more than 60,000 gave more false positives thus decreasing the accuracy.

The model was coded in Google Colab notebook and was trained on it utilizing the free tier(max 12hours of training).

- **Model Implementation Framework:** Detectron 2

- **Model:** Masked RCNN

- **Hyperparameters:** default hyperparameter selection done by the implementation

- **Train/Validation split of total data source:** 80%-20%

- **Learning Rate:** 0.005

- **Batch Size:** 512

- **Epochs:** 60,000 

## Step5: Storing the Model

The model was stored in the Google Drive. During the training the model is configured in such a way that after each 10,000 epochs of training the model is check-pointed into the google drive. We use the Detectron2 native way for storing the model.
### My interests:
* Machine Learning
* Deep Learning
* Computer Vision
* MLOps
* Dockers
* Python for Data Dcience


<!-- <hr>

<hr> -->

#### Feel free to connect with me at asit.piri@gmail.com or reach me at any of the below social channels

[<img align="left" alt="asit-piri | LinkedIn" width="30px" src="https://img.icons8.com/color/48/000000/linkedin.png" />][linkedin]
[<img align="left" alt="asit-piri | Twitter" width="30px" src="https://img.icons8.com/fluent/48/000000/twitter.png" />][twitter]
[<img align="left" alt="asit-piri | YouTube" width="30px" src="https://www.vectorlogo.zone/logos/youtube/youtube-tile.svg" />][YouTube]
[<img align="left" alt="asit-piri | Pexels" width="80px" src="http://images.pexels.com/lib/api/pexels.png" />][Pexels]

<br>

#### Website:https://asitpiri.dev/

<hr>

[linkedin]: https://www.linkedin.com/in/asit-piri-7128a510/
[twitter]: https://twitter.com/AsitPiri
[Pexels]: https://www.pexels.com/@asit-piri-260326689/
[YouTube]: https://www.youtube.com/user/asitpiri

<p align="left"> <img src="https://komarev.com/ghpvc/?username=asit-piri&label=Profile%20views&color=0e75b6&style=flat" alt="asit-piri" /> </p>

<p><img align="center" src="https://github-readme-streak-stats.herokuapp.com/?user=asit-piri&" alt="asit-piri" /></p>

=======
# Vehicle-Dent-Detection
>>>>>>> d9877f1 (committing)


## Create a new repository on the command line
1. echo "# reviewscrapper" >> README.md

2. git init

3. git add README.md

4. git commit -m "first commit"

5. git branch -M main

6. git remote add origin https://github.com/asit-piri/VehicleDentDetection.git

7. git push -u origin main

## Push an existing repository from the command line
1. git remote add origin https://github.com/asit-piri/VehicleDentDetection.git

2. git branch -M main

3. git push -u origin main 
