from actor_encoder import ActorEncoder
from one_hot_encoder import OneHotEncoderWrapper
from genre_encoder import GenreEncoder
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class MoviePipeline:
    def __init__(self, **kwargs):
        self.ohe = OneHotEncoderWrapper(handle_unknown='ignore')
        self.ge = GenreEncoder()
        self.ae = ActorEncoder(n_actors=9)
        self.sc = StandardScaler()
        self.model = XGBRegressor(**kwargs)

    def fit(self, X, y):
        X = self._preprocess(X)
        X = self.ae.fit_transform(X, y)
        X = self.ge.fit_transform(X)
        X = self.ohe.fit_transform(X)
        X = X.select_dtypes(['number'])
        cols = X.columns.to_list()
        # X = self.sc.fit_transform(X)
        self.model.fit(X, y)

        self.feature_importance = (
            pd.DataFrame(
                list(zip(cols, self.model.feature_importances_)),
                columns=['feature', 'importance'])
            .sort_values('importance', ascending=False)
        )

# From Towards Data Science to 
    def _preprocess(self, X):
        X = X.copy()
        X['year'] = X['date'].dt.year
        X['month_x'] = (np.sin(2 * np.pi * X['date'].dt.month/12)+1)/2
        X['month_y'] = (np.cos(2 * np.pi * X['date'].dt.month/12)+1)/2
        X['day_x'] = (np.sin(2 * np.pi * X['date'].dt.day/X['date'].dt.days_in_month)+1)/2
        X['day_y'] = (np.cos(2 * np.pi * X['date'].dt.day/X['date'].dt.days_in_month)+1)/2
        X['dow_x'] = (np.sin(2 * np.pi * X['date'].dt.day_of_week/7)+1)/2
        X['dow_y'] = (np.cos(2 * np.pi * X['date'].dt.day_of_week/7)+1)/2
        return X

    def _transform(self, X):
        X = X.copy()
        X = self._preprocess(X)
        X = self.ae.transform(X)
        X = self.ge.transform(X)
        X = self.ohe.transform(X)
        X = X.select_dtypes(['number'])
        # X = self.sc.transform(X)
        return X

    def predict(self, X):
        X = self._transform(X)
        return self.model.predict(X)

    def score(self, X, y):
        X = self._transform(X)
        return self.model.score(X, y)

    def inference(self, date, genre, cast,
                  lang, budget, country):
        df = pd.DataFrame(
            {
                'date': [date],
                'genre': [genre],
                'cast': [cast],
                'lang': [lang],
                'budget': [budget],
                'country': [country]
            }
        )
        df['date'] = pd.to_datetime(df['date'])
        return float(self.predict(df)[0])
    

