import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class OneHotEncoderWrapper(OneHotEncoder):
    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        super().fit(X[['lang', 'country']])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(super().transform(
            X[['lang', 'country']]).toarray(), columns=self.get_feature_names_out())
        return pd.concat([X.reset_index(), df], axis=1).set_index('index')

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
