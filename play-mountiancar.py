import tensorflow as tf
import numpy as np
import gym

# cartpole gives us 4 data points when env.reset() called
num_inputs = 2
# number of hidden neurons
num_hidden = 4
# only want 1 output (1 or 0, left or right)
num_outputs = 1

learning_rate = 0.1

initalizer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32,shape=[None, num_inputs])

hidden_layer_1 = tf.layers.dense(X,num_hidden,activation=tf.nn.elu,kernel_initializer=initalizer)

# logits are final most pure form of output before we squish down with sigmoid
logits = tf.layers.dense(hidden_layer_1,num_outputs)
# squish all the outputs into one output
outputs = tf.nn.sigmoid(logits)

# 1-prob of left move == prob of right move
probs = tf.concat(axis=1,values=[outputs,1-outputs])
# slightly randomly select an action based on the probabilties
action = tf.multinomial(probs,num_samples=1)

# convert action from tensor to float32
y = 1.0 - tf.to_float(action)

# tensorflow doing some calculus:
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
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
