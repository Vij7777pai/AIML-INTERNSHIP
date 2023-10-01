# Crop Price Prediction using LSTM

## Overview

This project aims to predict crop prices for the crops, Arecanut(Coca) and Coconut(Grade-I), in the Mangaluru market. The dataset spans from January 1st 2015 to December 31st 2022 and used an LSTM (Long Short-Term Memory) model for time series prediction.

## Using the Repository

To use this repository, you can clone it to your local machine using the following command:

```bash
git clone https://github.com/Vij7777pai/AIML-INTERNSHIP.git 
```

## Notebooks

- [Coca.ipynb](Coca.ipynb): Contains code for predicting Arecanut prices.
- [Grade-I.ipynb](grade-I.ipynb): Contains code for predicting Coconut prices.

## Installation

To run the notebooks, you'll need to install the following Python modules using pip:

```bash
pip install pandas numpy matplotlib seaborn keras tensorflow scikit-learn
```

## Dataset

The dataset is stored in Excel format (.xlsx) located in the Datasets folder. 
- ./Dataset/Coca/Coca(2015-2022).xlsx for Arecanut
- ./Dataset/Grade-I/GRADE-I(2015-2022).xlsx for Coconut

## Process

### Data Collection 

Collected historical crop price data for Arecanut and Coconut in the Mangaluru market from 01/01/2015 to 31/12/2022 from [here](https://www.krishimaratavahini.kar.nic.in/department.aspx).

### Data Preprocessing

* Loaded the excel file.
    * Changed it to pandas DataFrame
* Handled missing data, if any, by filling gaps or removing incomplete records.
* Explored the dataset to understand its characteristics.

### Feature Extracion

Extracted relevant features that can improve prediction accuracy such as Minimum, Maximum and Modal prices.

### Sequence Generation

* Generated sequences of data points for training, where each sequence contains a window of historical data that the LSTM model will use to make predictions.
    * In this project, we utilize an input sequence of 3 consecutive days' worth of historical crop price data, and our LSTM model leverages this information to make accurate predictions of future prices in the Mangaluru market 

### Model Architecture

Defined the LSTM model architecture using Keras:

```python
model = Sequential()
model.add(InputLayer((3,3)))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3))
```

### Model Training

Trained the LSTM model using the training dataset. Monitored performance on the validation set to prevent overfitting.

### Model Evaluation

Evaluated the model's performance on the test dataset using metrics such as mean absolute error (MAE) and mean squared error (MSE). Visualized predictions against actual prices.

### Saving the Best Model

The best-performing model is saved in the 'models' folder for future use.

### Saving the scaler objects 

Scaler objects used for data normalization are saved in the Scaler Objects folder. These scalers can be applied to new data before making predictions with the trained model.

### Hyperparameter Tuning

Experimented with different hyperparameter to optimize the model's performance.
    * Changing the input sequence
    * Adding new units, layers
    * learning rate, epochs

### Fine-tuning and Deployment

Once satisfied with the model's performance, fine-tuned it with the entire dataset. Deployed the trained model for making real-time predictions on streamlit.

To run the streamlit_deploy.py code, you'll need to install the streamlit module using pip:

```bash
pip install streamlit
```
Once after installing 

```bash
streamlit run streamlit_deploy.py
```




