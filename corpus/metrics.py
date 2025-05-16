from sklearn.metrics import ( 
    root_mean_squared_error, 
    mean_squared_error, 
    mean_absolute_error,
    mean_squared_log_error,
    root_mean_squared_log_error,

    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import matplotlib.pyplot as plt

class Performance:
    def __init__(self, y_true, y_pred):
        '''
        y_true (array like) : original label values
        y_pred (array like) : predicted label values
        '''
        self.y_true = y_true
        self.y_pred = y_pred

    def Cmatrix(self, plot=False, normalize=False, value_format='.0%'):
        if plot == True:
            ConfusionMatrixDisplay(self.y_true, self.y_pred, normalize=str(normalize), values_format=value_format)
            plt.show()

        return confusion_matrix(self.y_true, self.y_pred)
    
    def performance_score(self, score='pr', graph=False):
        match score:
            case 'p':
                return precision_score(self.y_true, self.y_pred)
            case 'r':
                return recall_score(self.y_true, self.y_pred)
            case 'pr' | 'rp':
                return (precision_score(self.y_true, self.y_pred),
                        recall_score(self.y_true, self.y_pred))
            case 'f1':
                return f1_score(self.y_true, self.y_pred)
            case 'roc' | 'auc':
                return roc_auc_score(self.y_true, self.y_pred)
            case _:
                raise ValueError('Unknown score type... use p,r,pr,f1,roc')
                


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