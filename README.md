# Deep Learning for Ovarian Cancer Subtype Classification (OCSC-NET)
This repository contains our Group 4s deep learning model for classifying Ovarian Cancer Subtypes using histophathological images.
OCSC-Net stands for Ovarian Cancer Subtype Classification Network
The purpose of this project was create a model that could aid in the early stage diagnosis of 5 different types of Ovarian Cancer 

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EffuUXAYi5BOgc6-eyp2eu8ByngPtJkDsogzcFgGPU8YXQ?e=gjgPYn) | [Slack channel](https://app.slack.com/client/T08645XD55G/C08778BPW9H) | [Project report](https://www.overleaf.com/project/677238aaa7c20ff32d5330d7) |
|-----------|---------------|-------------------------|


## Video Demo 
This is our full video demo

https://github.com/user-attachments/assets/9a15ef73-0a0d-47ba-9b22-be72afc9196f

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo
This a demo of our model running 
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

## 2. Installation
Copy the commands be low to create a conda enviornemnt and install the dependencies 
```bash
git clone https://github.com/sfu-cmpt340/2025_1_project_04
cd 2025_1_project_04
conda env create -f requirements.yml
conda activate amazing
conda activate CMPT_340_Environment
```

<a name="repro"></a>
## 3. Reproduction

### Execution via Conda Environment

Run the command below to run the model
```bash
src/complete_model_patched.py
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

### Execution via Manual Dependency Installations using Pip3 Python [recommended]

The best way to replicate our project is to leverage our python notebook.
The most up-to-date, patched version is located in the ```bash development``` directory.
Open the file pytorch_model_patched.ipynb in an IDE of your choice (ideally Google Colab).

Run each code block in sequence. Downloading the dataset will likely require you to enter your Kaggle username and key.
More information can be found here on how to get your Kaggle key: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md

Once your credentials have been verified, run the remaining code blocks that train the model and output visualization plots.

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
