import pandas as pd
from movie_pipeline import MoviePipeline
from sklearn.metrics import r2_score
from movie_data_loader import load_movie_data


from sklearn.model_selection import train_test_split

class RScore:
    df = load_movie_data.load_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('score', axis=1),
        df['score'],
        shuffle=False,
        test_size=.25)

    pipeline = MoviePipeline()
    pipeline.fit(X_train, y_train)
    # pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    print(f"Test R² Score: {test_score:.4f}")