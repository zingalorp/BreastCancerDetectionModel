import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(path):
    """Load raw data and drop redundant columns"""
    df = pd.read_csv(path)

    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def split_data(df, test_size=0.2, random_state=1):
    """Stratified split into training and testing sets"""
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

def balance_classes(X_train, y_train, method='smote', random_state=1):
    """Resample to address class imbalance"""
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=random_state)
    elif method == 'none':
        return X_train, y_train
    else:
        raise ValueError(f"Invalid method: {method}")
    return sampler.fit_resample(X_train, y_train)

def scale_features(X_train, X_test, method='standard'):
    """Scale features using the specified method"""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'none':
        return X_train, X_test
    else:
        raise ValueError(f"Invalid method: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled