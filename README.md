# Titanic-Survival-Prediction
This project builds a machine learning pipeline to predict passenger survival on the Titanic using the Kaggle Titanic Dataset.


## 📖 Overview
The goal is to predict which passengers survived the Titanic disaster using machine learning. The dataset includes passenger details such as age, sex, class, fare, and more. We use custom preprocessing, feature engineering, and a Random Forest classifier to make predictions.

## 📁 Dataset
1. train.csv: Training data with labels (survived or not).
2. test.csv: Test data for generating predictions.

## 🧹 Data Preprocessing
1. Mounted Google Drive and loaded CSV files using pandas.
2. Removed or imputed missing values (e.g., Age with mean).
3. Performed Stratified Shuffle Split for balanced train/test sampling.
4. Encoded categorical features using OneHotEncoder.
5. Dropped irrelevant columns like Name, Ticket, Cabin.

## 📊 Exploratory Data Analysis
1. Checked correlations using a heatmap.
2. Visualized distributions of Survived, Pclass, and Sex in both train and test sets.
3. Identified that features like Pclass, Sex, and Fare correlate with survival.

## ⚙️ Feature Engineering
Created custom transformers using scikit-learn:

1. AgeImputer: Imputes missing age values with the mean.
2. FeatureEncoder: One-hot encodes Embarked and Sex.
3. FeatureDropper: Removes unnecessary columns.

Combined transformers using a Pipeline.

## 🤖 Model Building
1. Used RandomForestClassifier as the base model.
2. Applied StandardScaler for normalization.
3. Performed GridSearchCV for hyperparameter tuning:

    i. n_estimators: 10, 100, 200, 500
   ii. max_depth: None, 5, 10
  iii. min_samples_split: 2, 3, 4

🔍 Evaluation
Achieved ~78.77% accuracy on the validation set.

Best model:
```
RandomForestClassifier(max_depth=5, n_estimators=500)
```
