from sklearn.metrics import ( 
    root_mean_squared_error, 
    mean_squared_error, 
    mean_absolute_error,
    mean_squared_log_error,
    root_mean_squared_log_error
)

def mean_error(y_true, y_pred, type='rmse'):
    '''
    y_true (array like) : original label values
    y_pred (array like) : predicted label values
    type (str) : rmse | mse | mae | msle | rmsle
    '''
    match type.lower():
        case 'rmse':
            return root_mean_squared_error(y_true, y_pred)
        case 'mse':
            return mean_squared_error(y_true, y_pred)
        case 'mae':
            return mean_absolute_error(y_true, y_pred)
        case 'msle':
            return mean_squared_log_error(y_true, y_pred)
        case 'rmsle':
            return root_mean_squared_log_error(y_true, y_pred)
        case _:
            raise ValueError("type of mean error is unrecognised... Use rmse, mse, mae")