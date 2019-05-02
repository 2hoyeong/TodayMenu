import pandas
import numpy as np
import tensorflow as tf
import nn.RNN as nw
import time
from pathlib import Path

한식          = 0
일식          = 1
중식          = 2
양식          = 3
패스트푸드    = 4
피자          = 5
치킨          = 6
분식          = 7
술집          = 8
카페          = 9

df = pandas.read_csv('E:/O/data.csv', engine='python')

savedir = './utility/data/trained'

matrix = df.values # dataframe to numpy
result = [[np.nan, np.nan, np.nan]] # 작업의 편의를 위해 배열의 모양을 잡아준다.

maxlen = 10

test = [0,0,0,0,0,0,0,0,0,0]
lena = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for line in matrix:
    if line[1] == " 한식":
        line[1] = 한식
    elif line[1] == " 일식":
        line[1] = 일식
    elif line[1] == " 중식 " or line[1] == " 중식":
        line[1] = 중식
    elif line[1] == " 양식":
        line[1] = 양식
    elif line[1] == " 패스트푸드":
        line[1] = 패스트푸드
    elif line[1] == " 피자":
        line[1] = 피자
    elif line[1] == " 치킨":
        line[1] = 치킨
    elif line[1] == " 분식":
        line[1] = 분식
    elif line[1] == " 술집":
        line[1] = 술집
    elif line[1] == " 카페":
        line[1] = 카페
    else:
        print(line)
    if test[line[1]] >= 2000:
        continue
    test[line[1]] += 1
    line[0] = line[0][:maxlen]
    w2v = np.zeros(maxlen)
    lena[len(line[0])] += 1
    for i in range(len(line[0])):
        w2v[i] = ord(line[0][i]) / 65535

    result = np.vstack((result, [w2v, line[1], line[0]]))

print(lena)
result = result[1:] # 배열의 모양을 잡기위한 [NaN, NaN, NaN]을 제거 한다.
np.random.shuffle(result)
data_length = int(len(result) * 0.8)
train_data = result[:data_length]
test_data = result[data_length:]

x_data = train_data[:, 0]
y_data = train_data[:, 1]
x_data = np.reshape(x_data, [-1, 1])
y_data = np.reshape(y_data, [-1, 1])

for i in range(len(x_data)):
    x_data[i][0] = x_data[i][0]

for i in range(len(y_data)):
    y_data[i][0] = np.eye(10)[y_data[i][0]]


data_length = int(len(result) * 0.8)

train_data = result[:data_length]
test_data = result[data_length:]

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 400
display_step = 200

# Network Parameters
num_input = maxlen # MNIST data input (img shape: 28*28)
timesteps = 1 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [None, timesteps, num_input])
Y = tf.placeholder(tf.int32, [None, num_classes])

keep_prob = tf.placeholder(dtype=tf.float32)

weights = {
    'l1' : tf.get_variable("w1", shape=[num_hidden, num_hidden], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
    'l2' : tf.get_variable("w2", shape=[num_hidden, num_hidden], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
    'l3' : tf.get_variable("w3", shape=[num_hidden, num_hidden], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
    'l4' : tf.get_variable("w4", shape=[num_hidden, num_hidden], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
    'out' : tf.get_variable("wout", shape=[num_hidden, num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'l1' : tf.get_variable("b1", shape=[num_hidden], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
    'l2' : tf.get_variable("b2", shape=[num_hidden], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
    'l3' : tf.get_variable("b3", shape=[num_hidden], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
    'l4' : tf.get_variable("b4", shape=[num_hidden], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
    'out' : tf.get_variable("bout", shape=[num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
}

def ARNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)


    cells = []

    # Define a lstm cell with tensorflow
    for _ in range(2):
        cells.append(tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, activation=tf.nn.relu)))
    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    
    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['l1']) + biases['l1']

logit = ARNN(X, weights, biases)
logits = tf.nn.dropout(tf.nn.relu(tf.matmul(logit, weights['l2']) + biases['l2']), keep_prob=keep_prob)
#logits = tf.nn.relu(tf.matmul(logit, weights['l2'] + biases['l2']))
logits = tf.nn.dropout(tf.nn.relu(tf.matmul(logit, weights['l3']) + biases['l3']), keep_prob=keep_prob)
logits = tf.nn.dropout(tf.nn.relu(tf.matmul(logit, weights['l4']) + biases['l4']), keep_prob=keep_prob)

logits = tf.matmul(logits, weights['out']) + biases['out']

prediction = tf.nn.softmax(logits=logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    for step in range(1, training_steps+1):
        #np.random.shuffle(train_data)
        a = time.time()
        #for data in train_data:
        for i in range(int(len(train_data) / batch_size)):
            batch_x = x_data[i*batch_size:(i + 1) * batch_size]
            batch_y = y_data[i*batch_size:(i + 1) * batch_size]
            batch_x = np.vstack([np.expand_dims([x[0]], 0) for x in batch_x])
            batch_y = np.vstack([np.expand_dims(x[0], 0) for x in batch_y])
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob : 0.7})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y, keep_prob : 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.1f}%".format(acc * 100))
        if step % 100 == 1:
            print("Time : {:.2f}s".format((time.time() - a)) + "\tRemain Time : {:.1f}seconds".format((time.time() - a) * (training_steps - step)))
    print("Optimization Finished!")
    r = 0
    w = 0
    ac = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nac = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for test in test_data:
        test_data = np.expand_dims(test[0], axis=0).reshape(1, 1, num_input)
        test_label = np.eye(10)[test[1]].reshape(1, 10)
        if int(sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob : 1.0})) == 1:
            print(test[2], test[1])
            r += 1
            ac[test[1]] += 1
        else:
            w += 1
            nac[test[1]] += 1
    print("[RESULT] R :", r ,"| W :", w)
    print("Testing Accuracy: ", ((r / ( r + w )) * 100))
    print(ac)
    print(nac)