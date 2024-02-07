from sklearn.metrics import mean_absolute_error
from numpy import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, max_error, \
    median_absolute_error, mean_absolute_error


def print_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Squared Error (MSE):              {mse:.10f}')

    rmse = sqrt(mse)
    print(f'Root Mean Squared Error (RMSE):        {rmse:.10f}')

    mae = mean_absolute_error(y_true, y_pred)
    print(f'Mean Absolute Error (MAE):             {mae:.10f}')

    r2 = r2_score(y_true, y_pred)
    print(f'R-squared (R²):                        {r2:.10f}')

    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.10f}')

    me = max_error(y_true, y_pred)
    print(f'Max Error (ME):                        {me:.10f}')

    med_ae = median_absolute_error(y_true, y_pred)
    print(f'Median Absolute Error (MedAE):         {med_ae:.10f}')