# Digit recognizer  with Random Forest

import csv

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def build_model():
    train_data = pd.read_csv('train.csv')

    y = train_data['label']
    X = train_data.iloc[:, 1:]
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.25)

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
    predictions = model.predict(test_data)

    with open('output.csv', 'w') as fout:
        writer = csv.writer(fout)
        write_output(writer, predictions)
