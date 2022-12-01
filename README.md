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

This project is an AI-powered app 🧠🤖 to estimate cost of construction projects in San Francisco. 

* [Presentation of project](https://docs.google.com/presentation/d/1uWvuKxi8LZJN_XV6F3pEtfRy1y2JgECC/edit?usp=sharing&ouid=117915938711430623839&rtpof=true&sd=true)

Some of the efforts include data cleaning, feature engineering, setting up machine learning models, predictive error calculation, parameter tuning, creating a dashboard and deploying an online app. 

Machine learning models were trained with the historical data coming from building permits of San Francisco available
since early 1980s (thanks to datasf.org). A set of parameters and machine learning models were tested (including Linear,Lasso model, E-Net, KRidge, GBoosting, XGBoost, LGBoost and Random Forest). **Random Forest Model**🌲🌳🌲🌳 was judged to use as a final model.

> Structural work cost 

> ✅ include cost of foundation, columns, beams, slabs, floors, roof and workmanship cost

> ❌ does not include land price, finishing work, electricity & plumbing and commercial costs

![pipeline](https://github.com/LHB-Group/Civil-Work-Bidding-And-Investment-Helper/blob/0b9bc8a0add95aa4bfb8555bd3746303d31c0cf0/.img_pipeline.PNG)

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
    ├── src                     # Scripts on functions 
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
