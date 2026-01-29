import pandas as pd


class MovieDataLoader:
    def __init__(self, filepath='imdb_movies.csv'):
        self.filepath = filepath
        self.df = None

    def load_preprocess(self):
        df = pd.read_csv(self.filepath)
        df = df[(df['status'] == ' Released')]
        df = (
            df
            .drop(['names', 'overview', 'orig_title', 'status', 'revenue'], axis=1)
            .rename({'date_x': 'date', 'orig_lang': 'lang', 'crew': 'cast', 'budget_x': 'budget'}, axis=1)
        )
        # Normalize the score
        df['score'] = df['score'] / 10
        # Process the cast by extracting each name in between commas
        df['cast'] = df['cast'].apply(lambda x: str(x).split(', ')[::2])
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        # Clean and split the genres
        df['genre'] = df['genre'].apply(lambda x: str(
            x).replace('\xa0', ' ')).str.split(', ')
        # Clean and get the first language of the movie
        df['lang'] = df['lang'].str.replace(
            ' ', '').apply(lambda x: x.split(',')[0])
        df = df.sort_values('date').reset_index(drop=True)
        self.df = df
        print(
            f"Loaded {df.shape[0]} movies that were released prior to 2024.")
        return df

    def get_data(self):
        """Return the preprocessed dataframe"""
        if self.df is None:
            raise ValueError(
                "Data not loaded. Call load_and_preprocess() first.")
        return self.df


load_movie_data = MovieDataLoader('imdb_movies.csv')

# Show a sample of the loaded movie data
df = load_movie_data.load_preprocess()
df.head()
