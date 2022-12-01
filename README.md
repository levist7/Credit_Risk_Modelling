# Credit Risk Modelling | Calculation of PD, LGD, EDA and EL with Machine Learning in Python  

In development

![python-shield](https://forthebadge.com/images/badges/made-with-python.svg)

## Table of contents
* [Background](#background)
* [Project](#project)
* [Key documents](#key-documents)
* [Technologies](#technologies)
* [Datasets](#datasets)
* [Getting Started](#getting-started)
* [Top-directory layout](#top-directory-layout)
* [License](#license)
* [Author](#author)
* [Contact](#contact)

## Background

San Francisco is a hyper-popular city with homeless community (20% of population), natural disaster risks and astronomical housing prices. Affordable housing in San Francisco has not been an option. Meanwhile, new affordable construction projects are high in need. Many investors consider construction projects to invest in SF, which can provide high return rate. For construction projects, engineers struggle to predict the construction project cost as reasonable as possible to win biddings. 🏗️ 💸

## Project

This project is an AI-powered project to model the credit risk in compliance with the Basel accords.

Credit risk modeling is important for financial institutions. It represents the risk of borroer not being able to pay back the loan amount, credit card or other types of loans. In some cases, borrowers can pay only partial of the amount and the principal amount and interest amount are not paid. Both statistics and machine learning play an important role in handling big data and provide statistical modeling. In the recession of 2008,

## Key documents
	
Notebooks  
1 - A preprocessing notebook
2 - A notebook on selecting features for probability of default (PD) and modelling PD
3 - A notebook on checking population stability index
4 - A notebook on modelling loss given default (LGD), exposure at default (EAD) and Expected Loss (EL)

## Technologies

Project is created with:
* Python 3.8
* Jupyter Notebook 6.4.12
* Python libraries (see /requirements.txt)
* VSCode 1.71.2

## Datasets

The dataset comes from Lending Club. It is a large US peer-to-peer lending company. Different versions of this dataset existing, here the data was taken from a version available on (kaggle.com)[https://www.kaggle.com/wendykan/lending-club-loan-data/version/1] 

It contains all available data for more than 800,000 consumer loans issued from 2007 to 2015.

The data was then divided into two: one including data from 2007 to 2014 and another including data of 2015. Assuming that data from 2007 to 2014 are available at the moment when Expected Loss models are built. Then, the models are checked with more recent data (data from 2015) to evaluate whether the applications we have received after building the Probability of Default (PD) model have similar characteristics with the applications or not.  

## Getting Started

To run this project, 
1. Clone the repo:
   ```sh
   git clone https://github.com/LHB-Group/Civil-Work-Bidding-And-Investment-Helper.git
   ```
2. Install [packages](#technologies)

3. Install python libraries
   ```sh
   pip3 install -r requirements.txt
   ```
   
## Top-directory layout

    .
    ├── NoteBooks               # Jupyter notebooks on ....
    ├── src                     # Scripts on functions ...
    ├── LICENSE
    ├── README.md 
    └── requirements.txt

## License

Distributed under the MIT License. See LICENSE.txt for more information.

## Author

[levist7](https://github.com/levist7)

## Contact

...

---
Made with ❤️ in Paris
---
