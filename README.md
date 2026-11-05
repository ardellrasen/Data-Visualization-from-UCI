# Household Electric Power Consumption — EDA and CNN Forecasting

## Overview

This project explores the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

The primary objective is to understand household electricity consumption patterns through time-series visualization and statistical analysis. The analysis focuses on identifying temporal patterns, consumption characteristics, data quality issues, and autocorrelation that can support future time-series forecasting using a Convolutional Neural Network (CNN).

This repository contains the Exploratory Data Analysis (EDA) stage and a 1D CNN forecasting stage.

## Dataset

**Dataset:** Individual Household Electric Power Consumption

**Source:** UCI Machine Learning Repository

**Link:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Dataset characteristics:
- Measurement period: December 2006 – November 2010.
- Original sampling frequency: 1 minute.
- Location: One household in Sceaux, France.
- Main target variable: `Global_active_power` (kW).

The dataset contains measurements of household electricity consumption and several electrical quantities.

### Main Variables

| Variable | Description | Unit |
|---|---|---|
| Global_active_power | Global active power consumption | kW |
| Global_reactive_power | Global reactive power consumption | kW |
| Voltage | Average voltage | V |
| Global_intensity | Global current intensity | A |
| Sub_metering_1 | Energy sub-metering 1 | Wh |
| Sub_metering_2 | Energy sub-metering 2 | Wh |
| Sub_metering_3 | Energy sub-metering 3 | Wh |

## Exploratory Data Analysis

The EDA is organized into several analytical sections.

### 1. Time-Series Overview
- Original electricity consumption at 1-minute resolution.
- Visualization of the entire observation period.
- Detailed inspection of a one-week period.

### 2. Resampling Analysis
- Comparison between original 1-minute data and hourly average consumption.
- Visualization of consumption trends after hourly resampling.
- Evaluation of the effect of temporal aggregation.

### 3. Daily and Weekly Patterns
- Average hourly consumption throughout the day.
- Comparison between weekdays and weekends.
- Average electricity consumption across days of the week.
- Day-hour consumption heatmap.

### 4. Monthly and Seasonal Patterns
- Average consumption by calendar month.
- Monthly consumption trends throughout the observation period.
- Initial investigation of long-term temporal variations.

### 5. Statistical Distribution
- Distribution of original 1-minute consumption data.
- Distribution of hourly average consumption.
- Mean and median comparison.
- Boxplots for hourly, daily-hourly, and monthly consumption.

### 6. Missing Data Analysis
- Percentage of missing observations per day.
- Identification of consecutive missing-data intervals.
- Investigation of data continuity issues.

### 7. Temporal Dependency Analysis
- Autocorrelation Function (ACF) analysis.
- Investigation of daily and weekly periodicity.
- Rolling mean and rolling standard deviation analysis.

## Key Observations

Based on the exploratory visualizations, several preliminary observations can be identified:

1. **Daily consumption patterns:** Electricity consumption exhibits recurring variations throughout the day, with higher consumption generally occurring during morning and evening periods.

2. **Weekly variations:** Weekend consumption patterns differ from weekday patterns.

3. **Temporal periodicity:** The ACF visualization indicates recurring autocorrelation patterns, particularly around 24-hour and 168-hour intervals.

4. **Data distribution:** Electricity consumption exhibits a positively skewed distribution, indicating that lower consumption values occur more frequently than high-consumption values.

5. **Missing observations:** Several periods contain missing data, including some relatively long consecutive gaps that require appropriate preprocessing before model development.

6. **Long-term variations:** Monthly and rolling statistics indicate that consumption characteristics vary throughout the observation period.

These findings provide an initial understanding of the dataset and informed the forecasting experiments below.

## CNN Forecasting

`cnn_forecast.py` forecasts the next hour of `Global_active_power` with a 1D CNN and compares it against simple baselines on a one-year test period.

| Model | Test MAE (kW) | Test RMSE (kW) |
|---|---|---|
| 1D CNN (24-hour window) | 0.355 | 0.526 |
| Persistence (t-1) | 0.417 | 0.629 |
| Seasonal naive (t-24) | 0.572 | 0.841 |
| Seasonal naive (t-168) | 0.585 | 0.846 |

## How to Run

Download the dataset ZIP from the link above into this folder, then run the commands below in PowerShell. Keep the virtual environment on a short path, because PyTorch can fail to install on long Windows paths.

```powershell
python -m venv "$env:USERPROFILE\power-venv"
$py = "$env:USERPROFILE\power-venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
& $py eda.py "individual+household+electric+power+consumption.zip" outputs
& $py cnn_forecast.py "individual+household+electric+power+consumption.zip" outputs
```

Figures and metrics are written to `outputs/`.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- statsmodels
- Keras (with the PyTorch backend)
- PyTorch

## Project Status

**Current stage:** EDA and CNN forecasting are complete.

Possible next steps include longer forecast horizons, repeated runs with different seeds, and better handling of consumption peaks.

## Reference

Hebrail, G., & Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository.

https://doi.org/10.24432/C58K54
