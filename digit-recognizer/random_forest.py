# https://www.kaggle.com/c/digit-recognizer

import csv

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@click.group()
def cli():
    pass


@cli.command()
@click.argument('output_filename')
def default(output_filename):
    X, y = load_data()
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    model = build_model(train_X, train_y)
    test_data = pd.read_csv('test.csv')
    predictions = model.predict(test_data)
    write_output(predictions, output_filename)


# Score = 0.93671
@cli.command()
@click.argument('output_filename')
def lines(output_filename):
    """Captures horizontal and vertical lines as features."""
    X, y = load_data()
    X = append_extra_features(X)
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    model = build_model(train_X, train_y)
    test_data = pd.read_csv('test.csv')
    test_data = append_extra_features(test_data)
    predictions = model.predict(test_data)
    write_output(predictions, output_filename)


@cli.command()
def feature_importances():
    X, y = load_data()
    X = append_extra_features(X)
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    model = build_model(train_X, train_y)
    import pdb; pdb.set_trace()
    importances = model.feature_importances_


def load_data(filename='train.csv'):
    train_data = pd.read_csv(filename)

    y = train_data['label']
    X = train_data.iloc[:, 1:]

    return X, y


def append_extra_features(X):
    vertical_lines = [X.iloc[:, i::28].mean(axis=1) for i in range(28)]
    horizontal_lines = \
        [X.iloc[:, i * 28:(i + 1) * 28].mean(axis=1) for i in range(28)]

    return pd.concat([X] + vertical_lines + horizontal_lines, axis=1)


# Default random forest model scored 0.93600 on Kaggle
def build_model(train_X, train_y):
    model = RandomForestClassifier(random_state=0)
    model.fit(train_X, train_y)

    return model


def write_output(predictions, filename='output.csv'):
    with open(filename, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['ImageId', 'Label'])
        for i, p in enumerate(predictions, 1):
            writer.writerow([i, p])


if __name__ == '__main__':
    cli()
