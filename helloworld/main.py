import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt


# Mostly copied from https://medium.com/@saxenarohan97/intro-to-tensorflow-solving-a-simple-regression-problem-e87b42fd4845


def main():
    # Get the data
    total_x = np.array([[float(x)] for x in range(50)])
    total_y = np.array([float(x) for x in range(50)])

    # Keep some samples for training
    train_x = total_x[:30]
    train_y = total_y[:30]

    # Keep some samples for validation
    valid_x = total_x[30:40]
    valid_y = total_y[30:40]

    # Keep remaining samples as test set
    test_x = total_x[40:]
    test_y = total_y[40:]

    w = tf.Variable(tf.truncated_normal([1, 1], mean=0.0, stddev=1.0,
                    dtype=tf.float64))
    b = tf.Variable(tf.zeros(1, dtype = tf.float64))

    y, cost = calc(train_x, train_y, w, b)

    # Feel free to tweak these 2 values:
    learning_rate = 0.000001
    epochs = 10000
    points = [[], []] # You'll see later why I need this

    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(init)
    # import pdb; pdb.set_trace()

        for i in list(range(epochs)):
            sess.run(optimizer)

            if i % 10 == 0.:
                points[0].append(i+1)
                points[1].append(sess.run(cost))

            if i % 100 == 0:
                print(sess.run(cost))

        print('Predictions = ', sess.run(y))
        print('w = {0}, b = {1}'.format(sess.run(w), sess.run(b)))

        # plt.plot(points[0], points[1], 'r--')
        plt.plot(train_x, train_y, 'ro', label='Training data')
        plt.plot(train_x, sess.run(w) * train_x + sess.run(b),
                 label='Linear model')
        # plt.axis([0, epochs, 0, 1000])
        plt.axis([0, 50, 0, 50])
        plt.legend()
        plt.show()

        _, valid_cost = calc(valid_x, valid_y, w, b)

        print('Validation error =', sess.run(valid_cost))

        _, test_cost = calc(test_x, test_y, w, b)

        print('Test error =', sess.run(test_cost))


def calc(x, y, w, b):
    # Returns predictions and error
    predictions = tf.add(b, tf.matmul(x, w))
    error = tf.reduce_mean(tf.square(y - predictions))

    return [predictions, error]


if __name__ == '__main__':
    main()
