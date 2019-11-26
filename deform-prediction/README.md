# deform-prediction
## Predictive analysis and seasonal forecasting for ground deformation using LSTMs

Default setting:
- Each of the 32 LSTM networks is learnt iteratively with samples from 1,2,...,32 previous months, respectively.
- The flag `useGps` set to `True` or to `False` indicates whether to use GPS or InSAR data, respectively.

Output: 
- The outputs are the prediction errors for a set of regression metrics: Root Mean Square Error (RMSE), Explained Variance (VAR), Mean Absolute Error (MAE), Median Absolute Error (MDAE), Maximum Residual Error (MRE), and Coefficient of determination (R^2).

### Usage 1:
#### run `lstm_preds_months.py` to predict the ground deformation for one future time step over the last year period on the given location.

Specific settings:
- All LSTMs are learnt with past observations from 30 days.

This setting shows the ability of the LSTMs to **fit** the model to the learning data.

### Usage 2:
#### run `lstm_forecast_months.py` to forecast the ground deformation over the last year period on the given location. 

Specific settings:
- Each of the 32 LSTM networks is learnt iteratively with observations from additional previous months. For each LSTM, more future observations are predicted over the next year.
- The flag `earlySeason` set to `True` considers as the starting validation data the early stages from the previous season. Otherwise, `earlySeason` set to `False` considers as the starting validation data the last stages from the previous season.

This setting shows the ability of the LSTMs to **forecast** the model to the test data.
