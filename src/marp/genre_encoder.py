import numpy as np
import pandas as pd


class GenreEncoder:
    def fit(self, df: pd.DataFrame) -> None:
        genres = set()
        for row in df['genre']:
            for genre in row:
                genres.add(genre)
        genres.remove('nan')
        genres = list(genres)
        self.genres = genres

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        a = np.zeros((df.shape[0], len(self.genres)))
        for i, row in enumerate(df.iloc):
            for j, genre in enumerate(self.genres):
                if genre in row['genre']:
                    a[i, j] = 1
        df = df.reset_index().join(pd.DataFrame(a, columns=self.genres)).set_index('index')
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
