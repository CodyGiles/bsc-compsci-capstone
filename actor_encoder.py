import pandas as pd
import numpy as np

class ActorEncoder:
    def __init__(self, n_actors=3):
        self.n_actors = n_actors

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        df = X.join(y)
        actor_scores_df = (
            df[['cast', 'date', 'score']]
            .explode('cast')
        )

        actor_scores_df['avg_score'] = (
            actor_scores_df
            .groupby('cast')['score']
            .expanding()
            .mean()
            .reset_index(drop=True)
        )

        actor_scores_df = (
            actor_scores_df
            .drop('score', axis=1)
            .sort_values(['cast', 'date'])
            .rename({'cast': 'actor'}, axis=1)
        )

        actor_scores_df['actor'] = actor_scores_df['actor'].apply(lambda x: str(x))
        self.actor_scores_df = actor_scores_df


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for i in range(self.n_actors):
            df = df.reset_index()
            df[f'actor_{i}'] = df['cast'].apply(lambda x: str(x[i]) if len(x) > i else '~')
            df2 = pd.merge(df, self.actor_scores_df, left_on=f'actor_{i}', right_on='actor')
            df2 = df2[df2['date_x'] > df2['date_y']].sort_values(['actor', 'date_x']).groupby('index').last()[['avg_score']]
            df = df.set_index('index').join(df2)
            df[f'actor_{i}'] = df['avg_score']
            df = df.drop('avg_score', axis=1)

        df['actor_mean'] = 0
        df['n_scores'] = 0

        for i in range(self.n_actors):
            df['actor_mean'] += df[f'actor_{i}'].apply(lambda x: x if x >= 0 else 0)
            df['n_scores'] += df[f'actor_{i}'].apply(lambda x: 1 if x >= 0 else 0)

        df['actor_mean'] = np.divide(df['actor_mean'], df['n_scores'])

        for i in range(self.n_actors):
            df[f'actor_{i}'] = df[[f'actor_{x+i}' for x in range(self.n_actors-i)]].bfill(axis=1).iloc[:,0]

        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)