from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
import pandas as pd

# Creating Custom Pipeline
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, get_feature_names):
        self.get_feature_names = get_feature_names

    def fit(self, X, Y = None):
        return self

    def transform(self, X, Y = None):
        return X[self.get_feature_names].values

def make_full_pipeline(df):
    df = pd.read_csv('https://raw.githubusercontent.com/Athena75/IBM-Customer-Value-Dashboarding/main/data/Customer-Value-Analysis.csv', index_col = 'Customer')
    X = df.drop(['Response'], axis = 1)
    Y = df.Response.apply(lambda X: 0 if X == 'No' else 1)

    catgs = [var for var, var_type in X.dtypes.items() if var_type == 'object']
    numls = [var for var in X.columns if var not in catgs]

    # Defining the steps in the categorical piepline
    catg_pipeline = Pipeline([
        ('catg_selector', FeatureSelector(catgs)),
        ('one_hot_encoder', OneHotEncoder(sparse = False)),
        ])

    # Defining the steps in the numerical pipeline
    numl_pipeline = Pipeline([
        ('numl_selector', FeatureSelector(numls)),
        ('std_scaler', StandardScaler()),
    ])

    # Combining the numerical and categorical pipeline into one pipeline
    full_pipeline = FeatureUnion(transformer_list = [
        ('numl_pipeline', numl_pipeline),
        ('catg_pipeline', catg_pipeline),
    ])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)
    _ = full_pipeline.fit_transform(X_train)
    return full_pipeline