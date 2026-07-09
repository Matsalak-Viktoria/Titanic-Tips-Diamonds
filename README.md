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
- Perform Exploratory Data Analysis (EDA) to understand feature distributions and relationships with the target variable.
- Build machine learning pipelines for classification and regression tasks:
  - **Passenger survival prediction** - Predict whether a passenger survived based on passenger information.
  - **Tip amount prediction** - Predict the restaurant tip amount based on customer and bill information.
  - **Diamond price prediction** - Predict the price of a diamond based on its physical characteristics.
- Investigate how different preprocessing strategies (missing value imputation and feature scaling) affect KNN model performance. 
- Analyze experimental results to identify the most effective model configurations.
 
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
3. Model Training and Evaluation
   - Outer Cross-Validation
      - Train/Test split
      - Pipeline Setup
      - Inner Cross-Validation with GridSearchCV
         - Data Preprocessing (Imputation, Scaling, Encoding)
         - KNN Model Training
         - Best Hyperparameter Selection
      - Prediction on Unseen Test Data
      - Model Evaluation  
4. Result Analysis
   - Comparison of Different Preprocessing Strategies
   - Best Configuration Selection

## Technologies
 - Python
 - Pandas
 - NumPy
 - Scikit-learn
 - Matplotlib
 - Seaborn

## Methods
### Data Preprocessing

**Numerical features**:
- Missing value imputation (Mean / Median)
- Feature scaling (StandardScaler / MinMaxScaler)

**Categorical features**:
- Most frequent value imputation
- One-Hot Encoding

### Machine Learning Model
- K-Nearest Neighbors (KNN)

**Hyperparameters optimized**:
- Number of neighbors (k)
- Weight function (uniform, distance)

### Validation Strategy
**Nested Cross-Validation**:
- Outer CV for robust model evaluation
- Inner CV with GridSearchCV for hyperparameter optimization

### Evaluation Metrics
**Classification**:
- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1-score (Weighted)

**Regression**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

## Experimental Configurations
Different preprocessing combinations were tested:
- Mean Imputation + StandardScaler
- Median Imputation + StandardScaler
- Mean Imputation + MinMaxScaler
- Median Imputation + MinMaxScaler

## Results

The following figures summarize model performance under different preprocessing configurations for each prediction task.

### Model Performance Comparison for Each Prediction Task:

### 1. Passenger Survival Prediction (Classification)  

![Metrics](Passenger_survival_metrics.jpg)

All four KNN models demonstrated high and stable performance. The mean values of Accuracy, Precision, Recall, and F1-score differ only slightly, indicating balanced model performance without favoring any particular class. The low standard deviations further confirm the robustness of the results across different training and test splits. Models using StandardScaler (with both mean and median imputation) achieved better average performance than those using MinMaxScaler. Regarding the impact of the imputation strategy, the results also depend on the scaling method: with StandardScaler, median imputation achieved slightly better performance, whereas with MinMaxScaler, mean imputation produced slightly higher average metrics. This suggests that the choice of imputation and scaling strategies should be based on the data distribution and the presence of outliers. Considering the overall experimental results, the model configured with impute_strategy = median and scale_strategy = standard can be regarded as the optimal configuration, as it achieved the best combination of average performance and stability.

### 2. Tip Amount Prediction (Regression) 

![Metrics](Tip_size_metrics.jpg)

Based on the experimental results, all four KNN models demonstrated good performance. However, the models using StandardScaler achieved the highest accuracy and stability. They also produced lower RMSE and MAE values, as well as smaller standard deviations, than the models using MinMaxScaler. Both imputation strategies (mean and median) produced identical results within the same scaling method, suggesting that the choice between them should be based on the data distribution and the presence of outliers. Overall, the most effective configuration is StandardScaler combined with either imputation strategy.

### 3. Diamond Price Prediction (Regression)  

![Metrics](Diamond_price_metrics.jpg)

The experimental results showed that all four KNN models demonstrated similar performance; however, certain differences can be observed depending on the scaling method. Models using MinMaxScaler achieved lower average errors (RMSE and MAE), indicating more accurate predictions, but slightly lower stability due to higher standard deviations. In contrast, models using StandardScaler were more stable, although they were less accurate. Imputing missing values using either the mean or median produced identical results, indicating that the choice between them does not affect prediction quality. Overall, the most balanced configuration is MinMaxScaler combined with either imputation strategy.
  
