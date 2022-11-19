# Credit Risk Modelling | Calculation of PD, LGD, EDA and EL with Machine Learning in Python  

In development

**Data Source**

The dataset comes from Lending Club. It is a large US peer-to-peer lending company. Different versions of this dataset existing, here the data was taken from a version available on (kaggle.com)[https://www.kaggle.com/wendykan/lending-club-loan-data/version/1  ] 

It contains all available data for more than 800,000 consumer loans issued from 2007 to 2015.

The data was then divided into two: one including data from 2007 to 2014 and another including data of 2015. Assuming that data from 2007 to 2014 are available at the moment when Expected Loss models are built. Then, the models are checked with more recent data (data from 2015) to evaluate whether the applications we have received after building the Probability of Default (PD) model have similar characteristics with the applications or not.  

**Notebooks**

1 - A preprocessing notebook
2 - A notebook on selecting features for probability of default (PD) and modelling PD
3 - A notebook on checking population stability index
4 - A notebook on modelling loss given default (LGD), exposure at default (EAD) and Expected Loss (EL)
