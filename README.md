# Passenger Survival, Tip Size, Diamond Price Prediction | [view code](https://github.com/Matsalak-Viktoria/Titanic-Tips-Diamonds/blob/main/Titanic_Tips_Diamonds.ipynb)

## Overview
This project explores the implementation and evaluation of machine learning pipelines for solving three different prediction tasks using real-world datasets.

The main goal of the project is not only to build predictive models, but also to investigate how different data preprocessing techniques influence the performance of K-Nearest Neighbors (KNN) models.

The project focuses on the following prediction problems:
- Titanic Dataset - Predicting passenger survival (classification)
- Tips Dataset - Predicting tip amount (regression)
- Diamonds Dataset - Predicting diamond price (regression)

For each task, multiple experiments were conducted using different preprocessing strategies and nested cross-validation for hyperparameter optimization and reliable model evaluation in order to compare predictive performance, robustness, and stability across different configurations.

## Objectives

The main objectives of this project are:
- Build machine learning pipelines for solving classification and regression tasks.
- Investigate how different preprocessing strategies (missing value imputation and feature scaling) influence KNN model performance.
- Analyze and compare experimental results to identify the most effective configurations.
 
## Datasets

### Titanic Dataset
- **Survived**: Target variable. Indicates whether the passenger survived (0 = No, 1 = Yes).
- **Pclass**: Passenger ticket class (1 = First, 2 = Second, 3 = Third). Represents socio-economic status.	
- **Sex**: Passenger gender (male / female).
- **Age**: Age of the passenger in years.
- **Sibsp**: Number of siblings or spouses traveling with the passenger aboard the ship.
- **Parch**: Number of parents or children traveling with the passenger aboard the ship.
- **Fare**: Ticket fare paid by the passenger.
- **Embarked**: Port of embarkation where the passenger boarded the ship (C = Cherbourg, Q = Queenstown, S = Southampton).
- **Class**: Text representation of passenger class (First, Second, Third). Contains the same information as pclass.
- **Who**: Passenger category based on age and gender (man, woman, child). Derived from sex and age.
- **Adult_male**: Indicates whether the passenger is an adult male (True / False). Derived from sex and age.
- **Deck**: Deck (section of the ship) where the passenger’s cabin was located. Contains many missing values.
- **Embark_town**: Full name of the embarkation location (Cherbourg, Queenstown, Southampton). Duplicate of embarked.
- **Alive**: Text version of survival status (yes / no). Duplicate of target variable survived, causing data leakage.
- **Alone**: Indicates whether the passenger was traveling alone (True / False). Can be inferred from sibsp and parch.

### Tips Dataset
- **Total_bill**: Total restaurant bill amount before the tip.
- **Tip**: Target variable. Tip amount left by the customer.
- **Sex**: Gender of the customer (male / female).
- **Smoker**: Indicates whether the customer is a smoker (Yes / No).
- **Day**: Day of the week when the visit occurred (Thur, Fri, Sat, Sun).
- **Time**: Time of day when the meal took place (Lunch / Dinner).
- **Size**: Number of people in the dining party.

### Diamonds Dataset
- **Carat**: Weight of the diamond. Usually the strongest factor affecting price.
- **Cut**: Quality of the diamond cut (Fair, Good, Very Good, Premium, Ideal).
- **Color**: Diamond color grade from D (best) to J (worst).
- **Clarity**: Measure of internal imperfections (I1 → IF, where IF is best).
- **Depth**: Total depth percentage of the diamond.
- **Table**: Width of the top surface relative to the widest point of the diamond.
- **Price**: Target variable. Price of the diamond in US dollars.
- **X**: Length of the diamond in millimeters.
- **Y**: Width of the diamond in millimeters.
- **Z**: Depth (height) of the diamond in millimeters.

## Workflow
The project workflow includes:
1. Exploratory Data Analysis (EDA)  
2. Feature Selection (only for Titanic dataset)  
3. Outer Cross-Validation  
   - Train/Test split
   - Pipeline Construction
   - Prediction on test fold
   - Performance Evaluation
4. Inner Cross-Validation with GridSearchCV (Hyperparameter Optimization)
   - Data Preprocessing (Imputation, Encoding, Scaling)
   - Hyperparameter Tuning
   - KNN Model Training 
5. Result Analysis (Comparison of Different Preprocessing Strategies)
   
The project workflow includes:
1. Exploratory Data Analysis (EDA)  
2. Feature Selection (only for Titanic dataset) 
3. Pipeline Construction
4. Outer Cross-Validation Split  
5. Inner Cross-Validation with GridSearchCV (Hyperparameter Optimization)   
5. Data Preprocessing (Imputation, Encoding, Scaling)  
7. KNN Model Training and Prediction  
8. Performance Evaluation    
9. Result Analysis and Comparison of Different Preprocessing Strategies

1. Data Analysis and Preparation  
2. Feature Selection  
3. Pipeline Construction  
4. Outer Cross-Validation Split  
5. Inner Cross-Validation with GridSearchCV  
6. Data Preprocessing and Model Training  
7. Prediction Generation  
8. Performance Evaluation  
9. Comparison of Different Preprocessing Strategies

1. Data Exploration  
2. Data Cleaning and Feature Selection  
3. Pipeline Construction  
4. Nested Cross-Validation Setup  
5. Preprocessing and Feature Transformation  
6. Hyperparameter Tuning with GridSearchCV  
7. KNN Model Training and Prediction  
8. Performance Evaluation  
9. Experimental Comparison
 
## Technologies
 - Python
 - Pandas
 - NumPy
 - Scikit-learn
 - Matplotlib
 - Seaborn

## Methods

## Evaluation Metrics
### Classification
- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1-score (Weighted)
### Regression
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

## Results
  
