#2024-08-Objective-identification-on-mental-health_CNN_LSTM

This project aims to use objective movement and physiology predictors to predict mental health by adopting deep learning (i.e., CNN-RNN and CNN-LSTM) approach. 

preprocessing.py: this .py file contains function for data processing

dataprocessing.ipynb: this file is the main file for data preprocessing. The output is "select_survey.csv" and "fitbit_survey.csv". These two datasets will be used in the main.ipynb.

main.ipynb: this file is the main file for deep learning analysis, which includes padding, training and testing data split, class of two models, hyperparameters tuning, and model evluation.