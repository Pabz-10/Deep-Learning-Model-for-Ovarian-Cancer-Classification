# Deep Learning for Ovarian Cancer Subtype Classification (OCSC-NET)
This repository is a template for your CMPT 340 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

This repository contains our final deep learning model for classifying Ovarian Cancer Subtypes using histophathological images.
OCSC-Net stands for Ovarian Cancer Subtype Classification Network
The purpose of this project was create a model that could aid in the early stage diagnosis of 5 different types of Ovarian Cancer 

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EffuUXAYi5BOgc6-eyp2eu8ByngPtJkDsogzcFgGPU8YXQ?e=gjgPYn) | [Slack channel](https://app.slack.com/client/T08645XD55G/C08778BPW9H) | [Project report](https://www.overleaf.com/project/677238aaa7c20ff32d5330d7) |
|-----------|---------------|-------------------------|


## Video Demo

https://github.com/user-attachments/assets/fe205129-2cbc-491b-a232-7e47aac4ecd6


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate CMPT_340_Environment
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
Run src/complete_model.py
Enter Kaggle Username and Key
```
Data can be found at ...
Output will be saved in output/models

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
