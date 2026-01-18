import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("Movie Rating Prediction")
    print("=======================")

    # 1. Load dataset
    data = pd.read_csv("imdb_Movies.csv")

    # 2. Keep required columns
    data = data[
        ['Name', 'Duration', 'Genre', 'Votes', 'Director',
         'Actor 1', 'Actor 2', 'Actor 3', 'Rating']
    ]

    # 3. Clean numeric columns
    data['Duration'] = data['Duration'].astype(str).str.replace(' min', '')
    data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')

    data['Votes'] = data['Votes'].astype(str).str.replace(',', '')
    data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')

    data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

    # 4. Encode categorical columns
    encoder = LabelEncoder()
    categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

    for col in categorical_cols:
        data[col] = data[col].astype(str)
        data[col] = encoder.fit_transform(data[col])

    # 5. Remove missing values
    data.dropna(inplace=True)

    # 6. Split features and target
    X = data.drop(['Rating', 'Name'], axis=1)
    y = data['Rating']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 8. Evaluate model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Mean Squared Error:", round(mse, 3))
    print("R2 Score:", round(r2, 3))

    # 9. Pick one movie from dataset
    movie_index = X_test.index[0]
    movie_features = X_test.loc[[movie_index]]
    actual_rating = y_test.loc[movie_index]
    predicted_rating = model.predict(movie_features)[0]

    movie_details = data.loc[movie_index]

    print("\nMovie Details")
    print("-------------")
    print("Movie Name :", movie_details['Name'])
    print("Duration   :", movie_details['Duration'], "minutes")
    print("Votes      :", int(movie_details['Votes']))
    print("Director   :", movie_details['Director'])
    print("Actor 1    :", movie_details['Actor 1'])
    print("Actor 2    :", movie_details['Actor 2'])
    print("Actor 3    :", movie_details['Actor 3'])

    print("\nRating Prediction")
    print("-----------------")
    print("Actual Rating   :", round(actual_rating, 2))
    print("Predicted Rating:", round(predicted_rating, 2))


if __name__ == "__main__":
    main()