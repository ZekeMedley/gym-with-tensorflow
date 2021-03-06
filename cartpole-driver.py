# OVERVIEW:
# 1-play several episodes of games
# 2-compute gradients
# 3-compute each actions score
# 4-multiply gradient vec by action score
# 5-calculate mean of gradient vec
# 6-train model on gradient vec

import tensorflow as tf
import gym
import numpy as np

print('finished imports')

def helper_discount_rewards(rewards, discount_rate):
    # takes in a list of rewards and applies the discount rate
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    # takes in all rewards, applies helper_discount function to apply discount,
    # and then normalizes results using mean and stddev.
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards,discount_rate))

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

# cartpole gives us 4 data points when env.reset() called
num_inputs = 4
# number of hidden neurons
num_hidden = 4
# only want 1 output (1 or 0, left or right)
num_outputs = 1

learning_rate = 0.01

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
env = gym.make('CartPole-v0')

num_game_rounds = 10
max_game_steps = 1000
num_iterations = 1500
# play with this
# determines how much the program discounts rewards from actions that
# occur after a certian action
# for example, w/0.9 future action rewards are reduced by *0.9
discount_rate = 0.8

print("starting session")

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_iterations):
        # string = 'on iteration ' + str(iteration)
        # printing is broken because windows is shit
        # print(string)

        all_rewards = []
        all_gradients = []

        for game in range(num_game_rounds):
            current_rewards = []
            current_gradients = []

            observations = env.reset()

            for step in range(max_game_steps):
                # uncomment to watch model train, no real reason to do this
                # env.render()
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X:observations.reshape(1,num_inputs)})

                observations, reward, done, info = env.step(action_val[0][0])

                current_rewards.append(reward)
                current_gradients.append(gradients_val)


                if done:
                    break

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards,discount_rate)
        feed_dict = {}

        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients

        sess.run(training_op,feed_dict=feed_dict)

    print("saving graph & sess")
    saver.save(sess, 'models/my-policy-gradient')

# 'play cartpole.py' has script for loading and running model
