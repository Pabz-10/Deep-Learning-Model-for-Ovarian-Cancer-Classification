# Deep Learning for Ovarian Cancer Subtype Classification (OCSC-NET)
This repository contains our final deep learning model for classifying Ovarian Cancer Subtypes using histophathological images.
OCSC-Net stands for Ovarian Cancer Subtype Classification Network
The purpose of this project was create a model that could aid in the early stage diagnosis of 5 different types of Ovarian Cancer 

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EffuUXAYi5BOgc6-eyp2eu8ByngPtJkDsogzcFgGPU8YXQ?e=gjgPYn) | [Slack channel](https://app.slack.com/client/T08645XD55G/C08778BPW9H) | [Project report](https://www.overleaf.com/project/677238aaa7c20ff32d5330d7) |
|-----------|---------------|-------------------------|


## Video Demo

https://github.com/user-attachments/assets/bc7a5418-799a-4a56-bdb3-2128006ffc87

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


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

## 2. Installation
Copy the commands be low to create a conda enviornemnt and install the dependencies 
```bash
git clone https://github.com/sfu-cmpt340/2025_1_project_04
cd 2025_1_project_04
conda env create -f requirements.yml
conda activate CMPT_340_Environment
```

<a name="repro"></a>
## 3. Reproduction
Run the command below to run the model
```bash
src/complete_model.py
## You will be prompted to enter your Kaggle Username and Key, which is required to download the dataset
## Key can be found in the settings tab of Kaggle (look under API and generate a new token, this token is your password)
```
Data can be found at .../data/train/ if using CUDA or\
./extracted-images/train/ if using Colab\
Output will be saved in output/models\

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
