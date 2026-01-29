from __future__ import annotations

import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from .movie_data_loader import MovieDataLoader
from .movie_pipeline import MoviePipeline


class ModelNotFitError(Exception):
    def __init__(self):
        super().__init__("Model must be fit first.")


"""
The model that tests and trains the input data.
"""


class Model:
    @property
    def test_score(self):
        if self.__getattribute__("score") is not None:
            return self.score
        raise ModelNotFitError

    @property
    def feature_importance(self):
        if self.__getattribute__("pipeline") is not None:
            return self.pipeline.feature_importance
        raise ModelNotFitError

    def fit(self, filepath: str) -> Model:
        load_movie_data = MovieDataLoader(filepath)
        df = load_movie_data.load_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('score', axis=1),
            df['score'],
            shuffle=False,
            test_size=.25
        )

        self.pipeline = MoviePipeline()
        self.pipeline.fit(X_train, y_train)
        self.score = self.pipeline.score(X_test, y_test)

        return self

    def inference(self, date, genre, cast, lang, budget, country) -> float:
        return self.pipeline.inference(date, genre, cast, lang, budget, country)

    def feature_importance_plot(self) -> None:
        # Extract feature importances
        df = (
            pd.DataFrame({
                'feature': self.pipeline.model.get_booster().feature_names,
                'importance': self.pipeline.model.feature_importances_
            })
            .sort_values(by='importance', ascending=False)
            .head(10)
        )

        # Plotting with Seaborn
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x='importance',
            y='feature',
            hue='feature',
            data=df,
            palette='viridis',
            legend=False
        )
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()


class MARP:
    def __init__(self, filepath: str) -> None:
        self.model = Model().fit(filepath)

    def predict(self, date, genre, cast, lang, budget, country) -> None:
        return self.model.inference(date, genre, cast, lang, budget, country)

    def user_prompt(self) -> None:
        """
        Prompt the user to input movie data and prints out the predicted audience score
        based on the trained model.
        """
        while True:
            inputs = self._get_inputs()
            output = self.model.inference(
                date=inputs['date'],
                genre=inputs['genre'],
                cast=inputs['cast'],
                lang=inputs['lang'],
                budget=inputs['budget'],
                country=inputs['country']
            )
            print(
                "Your movie will have an estimated audience/user score of: "
                f"{round(output, 1)}")
            break

    def _validate_date(self, date_str):
        try:
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def _validate_budget(self, budget_str):
        try:
            budget = float(budget_str.replace(',', '').replace('_', ''))
            return budget
        except ValueError:
            return None

        """
        The logic for inputting all the data from the user.
        """

    def _get_inputs(self):
        while True:
            date_input = input(
                "What is the release date of the movie? YYYY-MM-DD format: ")
            if not self._validate_date(date_input):
                print("Please enter in a date in YYYY-MM-DD format.")
                continue

            user_genres = input(
                "Genre of the movie? Can be multiple, separate with a comma e.g. "
                "Comedy, Action: ")
            genres = [s.strip() for s in user_genres.split(",")]

            cast_input = input(
                "Who's starring in the movie? Can be multiple, separate with a comma: ")
            cast = [s.strip() for s in cast_input.split(",")]

            lang_input = input("What language is the movie in? ")

            budget_input = input("What is the budget of the movie? ")
            budget = self._validate_budget(budget_input)
            if budget is None:
                continue

            country_input = input(
                "Where is the movie from? Use ISO 3166 country codes, e.g. US, AU, KR "
                "(Korea), etc.: ")

            return {
                'date': date_input,
                'genre': genres,
                'cast': cast,
                'lang': lang_input,
                'budget': budget,
                'country': country_input
            }


# v = pipeline.feature_importance
# v


# # Extract feature importances
# feature_importance_df = pd.DataFrame({
#     'feature': pipeline.model.get_booster().feature_names,
#     'importance': pipeline.model.feature_importances_
# })

# # Sort by importance
# feature_importance_df = feature_importance_df.sort_values(
#     by='importance', ascending=False).head(10)

# # Plotting with Seaborn
# plt.figure(figsize=(10, 8))
# sns.barplot(
#     x='importance',
#     y='feature',
#     hue='feature',
#     data=feature_importance_df,
#     palette='viridis',
#     legend=False
# )
# plt.title('Feature Importance')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()

# # %%
# # Generate predictions
# y_pred = pipeline.predict(X_test)

# # Create a DataFrame for easier plotting
# results_df = pd.DataFrame({
#     'Actual': y_test,
#     'Predicted': y_pred
# })

# # Scatter plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='Actual', y='Predicted', data=results_df, alpha=0.6)

# # Add a reference line y = x (perfect prediction)
# plt.plot([results_df['Actual'].min(), results_df['Actual'].max()],
#          [results_df['Actual'].min(), results_df['Actual'].max()],
#          color='red', linestyle='--')

# plt.title('Actual vs. Predicted User Scores')
# plt.xlabel('Actual User Scores')
# plt.ylabel('Predicted User Scores')
# plt.show()

# # %%
# sns.histplot(df['score'], bins=20)

# # %%
# # Generate predictions
# y_pred = pipeline.predict(X_test)

# # Calculate R² score
# r2 = r2_score(y_test, y_pred)

# print(f'R² score: {r2:.4f}')
