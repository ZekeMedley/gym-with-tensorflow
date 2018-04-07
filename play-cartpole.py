# load the CartPole ai and watch it play

import tensorflow as tf
import numpy as np
import gym

print("finished imports")

# re-define variables for saver

num_inputs = 4
num_hidden = 4
num_outputs = 1

learning_rate = 0.01

initalizer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32,shape=[None, num_inputs])

hidden_layer_1 = tf.layers.dense(X,num_hidden,activation=tf.nn.elu,kernel_initializer=initalizer)

logits = tf.layers.dense(hidden_layer_1,num_outputs)
outputs = tf.nn.sigmoid(logits)

probs = tf.concat(axis=1,values=[outputs,1-outputs])
action = tf.multinomial(probs,num_samples=1)

# convert action from tensor to float32
y = 1.0 - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

gradients_and_vars = optimizer.compute_gradients(cross_entropy)

gradients = []
gradient_placeholders = []
gradients_and_vars_feed = []

for gradient, var in gradients_and_vars:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32,shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    gradients_and_vars_feed.append((gradient_placeholder,var))

training_op = optimizer.apply_gradients(gradients_and_vars_feed)

saver = tf.train.Saver()

# try to run Session

env = gym.make("CartPole-v0")
observations = env.reset()

with tf.Session() as sess:
    saver.restore(sess, 'models/my-policy-gradient')

    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([action, gradients], feed_dict={X: observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])
