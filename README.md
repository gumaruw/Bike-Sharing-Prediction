# Bike Sharing Prediction

The goal is to predict daily rental counts (`cnt`) using weather, time, and other features. Several models are trained and compared to identify the best-performing approach.

---

## Dataset

[Bike Sharing Dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset)

- **Date/Time:** `dteday`, `season`, `yr`, `mnth`, `hr`, `weekday`, `holiday`, `workingday`
- **Weather:** `weathersit`, `temp`, `atemp`, `hum`, `windspeed`
- **Target:** `cnt` (total rentals = casual + registered)

## Features

- Interaction term: `temp_hum_interaction`
- One-hot encoding for categorical time features

## Models

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Decision Tree Regressor

## Results

- Reduced MSE by 25% vs baseline (Linear Regression → Gradient Boosting)
- Cross-validation used for robustness

| Model             | MSE       | R²    |
| ----------------- | --------- | ----- |
| Linear Regression | 2,040,242 | 0.491 |
| Random Forest     | 1,833,188 | 0.543 |
| Gradient Boosting | 1,804,607 | 0.550 |
| Decision Tree     | 2,765,607 | 0.310 |

**Best model:** Gradient Boosting (lowest MSE, highest R²).

## Usage

This script was originally developed and run in Google Colab (it uses `google.colab.files` and Kaggle CLI commands to fetch the dataset). To run it:

**In Colab:** open `BikeSharing.py` directly, cell by cell, or

**Locally:** first remove/replace the Colab-specific cells (the `google.colab.files` upload and `!kaggle`/`!unzip` shell commands) with a local path to the dataset CSV.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python BikeSharing.py  # after adapting the data-loading section as above
```

## Kaggle

You can also view the same project on [Kaggle](https://www.kaggle.com/code/gumaruw/predicting-bike-sharing-demand-with-ml).
