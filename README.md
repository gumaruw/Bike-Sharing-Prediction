# Bike Sharing Prediction

The goal is to predict daily rental counts (`cnt`) using weather, time, and other features. Several models are trained and compared to identify the best-performing approach.   

---

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)

---

## Dataset
[Bike Sharing Dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset/data)
- **Date/Time:** `dteday`, `season`, `yr`, `mnth`, `hr`, `weekday`, `holiday`, `workingday`  
- **Weather:** `weathersit`, `temp`, `atemp`, `hum`, `windspeed`  
- **Target:** `cnt` (total rentals = casual + registered)  

---

## Features
- Interaction term: `temp_hum_interaction`  
- One-hot encoding for categorical time features  

---

## Models
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Decision Tree Regressor  

---

## Results
| Model               | MSE        | R²    |
|----------------------|------------|-------|
| Linear Regression    | 2,040,242  | 0.491 |
| Random Forest        | 1,833,188  | 0.543 |
| Gradient Boosting    | 1,804,607  | 0.550 |
| Decision Tree        | 2,765,607  | 0.310 |

**Best model:** Gradient Boosting (lowest MSE, highest R²).  

---

## Usage
1. ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
2. ```bash
   python BikeSharing.py
(Ensure the dataset file is in the same directory or update the file path accordingly.)

## Kaggle
You can also access the same project on [Kaggle](https://www.kaggle.com/code/gumaruw/predicting-bike-sharing-demand-with-ml)
