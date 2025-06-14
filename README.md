# Deep Learning for Ovarian Cancer Subtype Classification (OCSC-NET)
This repository contains our Groups deep learning model for classifying Ovarian Cancer Subtypes using histophathological images.
OCSC-Net stands for Ovarian Cancer Subtype Classification Network
The purpose of this project was create a model that could aid in the early stage diagnosis of 5 different types of Ovarian Cancer 

## Project Report

You can view the report [here](Project%20Report.pdf).


## Video Demo 
https://github.com/user-attachments/assets/dcb5fe41-1ac0-490e-8ed3-200479497a40

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

<a name="demo"></a>
## 1. Example demo

https://github.com/user-attachments/assets/c208ae70-7716-449f-b4e0-6d0756c669ca

### What to find where
Overview of the folder directory

```bash
repository
├── development                  ## Contains jupyter Notebooks code of previous models
    ├── models                   ## Stores best model
├── output                       ## Stores confusion matrix andmodel plots for training loss and validation accuracy 
├── src                          ## Contains completed model 
├── README.md                    ## You are here
├── requirements.yml             ## Used to install required packages/dependencies
```

<a name="installation"></a>

## 2. Prerequisites

We have 1 .ipynb file and 2 python files available for project replication.
All 3 achieve the same results but through different setups and python versions.

```bash src/complete_model.py ``` is suitable if running our project in a Conda environment with Python version 3.12.4 or earlier

```bash src/complete_model_patched.py ``` is suitable if running our project with manual dependency installation using pip3 with Python version 3.13 or greater

```bash development/complete_model_patched.ipynb ``` is suitable for running our project via Google Colab or another sufficient IDE that runs .ipynb files

Why do we have 2 python files? Because Python 3.12.4 has deprecated the use of the od module's download API.
The difference in versions correspond to different API calls download the dataset.
3.12.4 can use od's download API. 3.13 requires an alternative solution via Kaggle's download API.


### Conda Installation [for running ```bash src/complete_model.py ```]
Copy the commands below to create a conda enviornemnt and install the dependencies 

```bash
git clone https://github.com/sfu-cmpt340/2025_1_project_04
cd 2025_1_project_04
conda env create -f requirements.yml
conda activate CMPT_340_Environment
```

### Python Installation [for running ```bash src/complete_model_patched.py ```]
If you plan on running our code using pip3 and python directly, we assume that you have python 3.13 installed on your machine (latest version).
To install Python, visit https://www.python.org/downloads/ and run their installer.


<a name="repro"></a>
## 3. Reproduction

### Execution via Conda Environment [requires Python 3.12.4 or earlier]

This method assumes that the conda environment is available on your machine.
CSIL machines have access to Conda, but require additional configuration overhead.

More information for accessing the Conda environment on a CSIL machine can be found here: https://coursys.sfu.ca/2023fa-cmpt-353-d1/pages/InstallingPython

Run the command below to run the model
```bash
cd src
python complete_model_patched.py
## You will be prompted to enter your Kaggle Username and Key, which is required to download the dataset
## Key can be found in the settings tab of Kaggle (look under API and generate a new token, this token is your password)
```
Data can be found at .../data/train/ if using CUDA or\
./extracted-images/train/ if using Colab\
Output will be saved in output/models

### Execution via Manual Dependency Installations using Pip3 Python [not recommended]
Alternatively, you can run the python file directly without a conda environment
```bash
pip3 install torch torchvision numpy tqdm matplotlib seaborn kagglehub scikit-learn
cd src
python complete_model_patched.py
```
However, this approach is not recommended on a CSIL machine due to restricted cache capacity. 
The dependencies alone will likely overload the cache. Further, the dataset requires an additional 3.4GB of cache space due to the nature of kagglehub's download API.

Why use kagglehub's download API? As of Python 3.13 has deprecated the use of the od module's download API.
To fully replicate our project, we recommend using the next approach, especially if running on a CSIL machine.

### Execution via Manual Dependency Installations using Pip3 Python [recommended!]

The best way to replicate our project is to leverage our python notebook.
The most up-to-date, patched version is located in the ```bash development``` directory.
Open the file pytorch_model_patched.ipynb in an IDE of your choice (ideally Google Colab).

Run each code block in sequence. Downloading the dataset will likely require you to enter your Kaggle username and key.
More information can be found here on how to get your Kaggle key: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md

Once your credentials have been verified, run the remaining code blocks that train the model and output visualization plots.
