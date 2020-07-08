import gym
import numpy as np
import random

env = gym.make("Pendulum-v0")

def get_discret_state(state):
	discrete_state = (state - env.observation_space.low) / STATE_VAL_STEP	
	return tuple(discrete_state.astype(np.int))

MAX_ACTION_VAL = env.action_space.high[0]
MIN_ACTION_VAL = env.action_space.low[0]
NB_ACTION = 8
ACTION_STEP = (MAX_ACTION_VAL - MIN_ACTION_VAL) / (NB_ACTION)

NB_DISCRET_VALUE = [20, 20, 20]
STATE_VAL_STEP = (env.observation_space.high - env.observation_space.low) / (NB_DISCRET_VALUE - np.array([1, 1, 1]))

EPSILON = 0.5
LEARNING_RATE = 0.1
FUTURE_DISCOUNT = 0.95

q_table = np.random.uniform(low=-0.1, high=0.1, size=(NB_DISCRET_VALUE + [NB_ACTION]))

for ep in range(600):
	av_reward = 0
	iteration = 3000
	if (EPSILON > 0):
		EPSILON -= 0.01
	current_state = env.reset()
	
	for i in range(iteration):

		discrete_state = get_discret_state(current_state)

		if random.uniform(0.0, 1.0) < EPSILON:
			action_index = random.randrange(0, NB_ACTION - 1)
		else:
			action_index = np.argmax(q_table[discrete_state])

		action = (action_index * ACTION_STEP) + MIN_ACTION_VAL

		new_state, reward, _, _ = env.step([action])
		av_reward += reward


		action_index = tuple(np.array([action_index]).astype(np.int))
		current_q = q_table[discrete_state + action_index]

		new_discrete_state = get_discret_state(new_state)
		max_future_q = np.max(q_table[new_discrete_state])

		q_table[discrete_state + action_index] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + FUTURE_DISCOUNT * max_future_q)
		current_state = new_state

		if (ep) % 100 == 0:
			env.render()

	print(f"Number of epoch: {ep}, average reward: {av_reward / iteration}")

env.close()