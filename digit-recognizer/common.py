import csv

import pandas as pd


def load_data(filename='train.csv'):
    train_data = pd.read_csv(filename)

    y = train_data['label']
    X = train_data.iloc[:, 1:]

    return X, y


def score(predicted_y, actual_y):
    comp = actual_y == predicted_y
    return 1 - (comp[comp == False].count() / actual_y.count())


def write_output(predictions, filename='output.csv'):
    with open(filename, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['ImageId', 'Label'])
        for i, p in enumerate(predictions, 1):
            writer.writerow([i, p])
