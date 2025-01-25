# Predicting Bike Sharing Demand with Machine Learning: A Comprehensive Data Science Project

This project aims to predict bike sharing demand using various machine learning models. The dataset used for this project is the Bike Sharing Dataset, which contains information about bike rentals over time, including weather conditions, date and time, and other relevant features.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Models](#models)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to predict the total count of bike rentals for a given day based on various features such as temperature, humidity, weather conditions, and more. Multiple machine learning models are trained and evaluated to determine the best model for this task.

## Dataset
The dataset used in this project is the [Bike Sharing Dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset/data) from Kaggle. It contains the following columns:
- `instant`: Record index
- `dteday`: Date
- `season`: Season (1: winter, 2: spring, 3: summer, 4: fall)
- `yr`: Year (0: 2011, 1: 2012)
- `mnth`: Month (1 to 12)
- `hr`: Hour (0 to 23)
- `holiday`: Whether the day is a holiday or not
- `weekday`: Day of the week (0: Sunday, 6: Saturday)
- `workingday`: Whether the day is a working day or not
- `weathersit`: Weather situation (1: Clear, 2: Mist, 3: Light Snow/Rain)
- `temp`: Normalized temperature in Celsius
- `atemp`: Normalized feeling temperature in Celsius
- `hum`: Normalized humidity
- `windspeed`: Normalized wind speed
- `casual`: Count of casual users
- `registered`: Count of registered users
- `cnt`: Count of total rentals (casual + registered)

## Features
The following features were used in the model:
- `temp`: Normalized temperature in Celsius
- `atemp`: Normalized feeling temperature in Celsius
- `hum`: Normalized humidity
- `windspeed`: Normalized wind speed
- `temp_hum_interaction`: Interaction term between temperature and humidity
- One-hot encoded features for month and day of the week

## Models
The following machine learning models were used:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Decision Tree Regressor

## Evaluation
Models were evaluated using the following metrics:
- Mean Squared Error (MSE)
- R^2 Score
- Cross-validation R^2 Score

## Data Visualization
Data visualizations were performed to understand the distribution and relationship of features with the target variable.

## Installation
To run this project, you need to have Python and the following libraries installed:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip.

## Usage
You can run the project by executing the script that contains the model training and evaluation code. Make sure to place the dataset file in the same directory as the script or update the file path accordingly.

## Results
The results of the model evaluation are as follows:

Linear Regression: MSE = 2040242.138, R^2 = 0.491
Random Forest: MSE = 1833188.785, R^2 = 0.543
Gradient Boosting: MSE = 1804607.662, R^2 = 0.550
Decision Tree: MSE = 2765607.205, R^2 = 0.310
The Gradient Boosting model performed the best with the lowest Mean Squared Error and highest R^2 Score.

## Kaggle
You can also access the same project on Kaggle using this link: (https://www.kaggle.com/code/gumaruw/predicting-bike-sharing-demand-with-ml)

## Contributing
Contributions are welcome! If you have any suggestions, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
