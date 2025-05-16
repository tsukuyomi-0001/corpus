from sklearn.model_selection import (
    train_test_split, 
    # StratifiedShuffleSplit,
)

def split(*array ,test_size=0.15, random_state=None, n_splits=1, type='SimpleSplit'):
    '''
    *array (array like): array of value of feature and label
    test_size (integer): size of test dataset
    random_state (integer): seed for random
    n_splits (integer): dataset split
    type (string): simple...
    '''
    match type.lower():
        case 'simple':
            return train_test_split(*array,  test_size=test_size, random_state=random_state)