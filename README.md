# Titanic Project

## Project Title
Predicting Titanic Survival based on Machine Learning Models

## Project Overview
The goal of this project is to create predictive machine learning models to predict whether or not a passenger survives the Titanic tragedy. Binary classification models such as Logistic Regression, Decision Trees, Random Forests, and XGBoost were trained and evaluated on the training dataset. A champion model was then selected to predict on the test dataset if a passenger survives or not.

## Table of Contents
- [The Challenge](#the-challenge)
- [Data Overview](#data-overview)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Missing Values](#missing-values)
  - [LabelEncoding Features](#labelencoding-features)
- [Feature Engineering](#feature-engineering)
- [Modelling](#modelling)
  - [Logistic Regression](#logistic-regression)
  - [Ensemble Learning](#ensemble-learning)
- [Selecting Champion Model](#selecting-champion-model)
- [Predicting on Test Dataset](#predicting-on-test-dataset)

## The Challenge
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

The challenge is to build a predictive model that answers the question: "what sorts of people were most likely to survive?" using passenger data (i.e. name, gender, socio-economic class, etc).

## Data Overview
- The ***train.csv*** file is the training dataset, we shall refer to it as `train_data`.
- `train_data` contains, 891 rows and 12 columns.
- The target variable is `Survived`, while the other variables are features.
  
Below shows a snapshot of the Data Dictionary, explain what each variables are:

![data_dictionary](https://github.com/justin-97/Titanic-Project/blob/main/Images/datadictionary.jpg)

Here's a preview of the first 10 rows of the training dataset:

![train dataset head](https://github.com/justin-97/Titanic-Project/blob/main/Images/traindatasethead.jpg)

## Exploratory Data Analysis
At a first glance of `train_data`, there appears to be missing values in 3 features:
 1) `Age`: 177 missing values.
 2) `Cabin`: 687 missing values.
 3) `Embarked`: 2 missing values.

A potential way we could deal with this missing values would be:
 1) `Age`: Replace the missing values with the aggregated mean ages of the total passengers.
 2) `Cabin`: Split `Cabin` into categorial values, group the missing values to an assigned term e.g. *'X'*
 3) `Embarked`: Considering we are using Ensemble Learning models (which are robust to outliers & missing values), we could just leave the missing values. 2 out of 891 is considered pretty insignificant.

#### 1) Age
Digging deeper into the `Age` variable, a boxplot was used to illustrate the distribution of ages of all passengers.

