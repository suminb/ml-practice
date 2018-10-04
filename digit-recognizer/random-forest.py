# https://www.kaggle.com/c/digit-recognizer

import csv

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_data(filename='train.csv'):
    train_data = pd.read_csv(filename)

    y = train_data['label']
    X = train_data.iloc[:, 1:]

    X = append_extra_features(X)

    # Returns train_X, val_X, train_y, val_y
    return train_test_split(X, y, test_size=0.25)


def append_extra_features(X):
    vertical_lines = [X.iloc[:, i::28].mean(axis=1) for i in range(28)]
    horizontal_lines = \
        [X.iloc[:, i * 28:(i + 1) * 28].mean(axis=1) for i in range(28)]

    return pd.concat([X] + vertical_lines + horizontal_lines, axis=1)


# Default random forest model scored 0.93600 on Kaggle
def build_model():
    train_X, val_X, train_y, val_y = load_data()

    model = RandomForestClassifier(random_state=0)
    model.fit(train_X, train_y)

    return model


def write_output(csv_writer, predictions):
    csv_writer.writerow(['ImageId', 'Label'])
    for i, p in enumerate(predictions, 1):
        csv_writer.writerow([i, p])


if __name__ == '__main__':
    model = build_model()
    test_data = pd.read_csv('test.csv')
    test_data = append_extra_features(test_data)
    predictions = model.predict(test_data)

    with open('output.csv', 'w') as fout:
        writer = csv.writer(fout)
        write_output(writer, predictions)
