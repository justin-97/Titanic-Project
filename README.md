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
First step, is to understand what the training data consists of. This is done by calling the `train_data.infdo()`. It can be seen below that the Dtype of the variables ranges from int64, object, and float64.

![train data info]()

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

**insert ages boxplot**

**insert histogram**

**Determining Interquartile Range (IQR) & no. of Outliers**

The IQR was determined by subtracting the 75th percentile and 25th percentile. The upper and lower limits were then calculated using the IQR value. Any values outside of these limits i.e. below the lower limit and/or above the upper limit shall be deemed outliers. It was determined that there were only 11 outliers in the `Age`.

We can leave these 11 passengers in, while we aggregate the `Age` mean of all passengers. This was determined as ***29.7*** years old. Lastly, we shall replace all missing values in the `Age` variable with 29.7.

#### 2) Cabin
Earlier, the `Cabin` column values were inconsistent. Cabins were assigned with a Letter, ranging from (A-H or T), followed with 1-3 numbers, indicating the room number.

