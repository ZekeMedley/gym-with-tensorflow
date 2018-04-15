import tensorflow as tf
import numpy as np
import gym


num_inputs = 2
num_outputs = 3
# play with this no idea what is good here:
num_hidden = 3

learning_rate = 0.1

num_game_rounds = 10
max_game_steps = 200
num_iterations = 250
# play with this
# determines how much the program discounts rewards from actions that
# occur after a certian action
# for example, w/0.9 future action rewards are reduced by *0.9
discount_rate = 0.8

# BUILDING MODEL

X = tf.placeholder(tf.float32, shape=[None, num_inputs])
actf = tf.nn.relu

initalizer = tf.contrib.layers.variance_scaling_initializer()

# concider playing with additional neurons/ layers later
hidden_layer_1 = tf.layers.dense(
                X,
                num_hidden,
                activation=actf,
                kernel_initializer=initalizer)

logits = tf.layers.dense(
        hidden_layer_1,
        num_outputs)

# apply activation function
outputs = tf.nn.sigmoid(logits)

# slightly randomly select an action based on the probabilties
action = tf.multinomial(outputs,num_samples=1)

# tensorflow doing some calculus:
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

# returns a list of the gradients and vars
gradients_and_vars = optimizer.compute_gradients(cross_entropy)

gradients = []
gradient_placeholders = []
gradients_and_vars_feed = []

# put everything togehter in a way that optimizer can understand
for gradient, var in gradients_and_vars:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32,shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    gradients_and_vars_feed.append((gradient_placeholder,var))

# train
training_op = optimizer.apply_gradients(gradients_and_vars_feed)

# init model and saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# make enviroment
env = gym.make('MountainCar-v0')
observations = env.reset()

print("starting session")

with tf.Session() as sess:
    saver.restore(sess, 'models/my-mountiancar')

    for x in range(1000):
        env.render()
        action_val, gradients_val = sess.run([action, gradients], feed_dict={X: observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])
