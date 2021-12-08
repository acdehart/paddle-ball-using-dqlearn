import random
from threading import Thread
from time import sleep, time

import numpy as np
from keras.engine.input_layer import InputLayer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


class paddle_boy:
    def __init__(self):
        self.start_time = time()
        self.actions = ['left', 'none', 'right']
        self.direction = "left"
        self.acceleration = 0
        self.velocity = 0
        self.goFlag = False
        self.hits = 0
        self.score = 0

        self.ball_distance_x = 0
        self.ball_distance_y = 0
        self.left_distance = 250
        self.right_distance = 250

        self.model = Sequential([
        # self.model.add(InputLayer(input_shape=(4,)))
            Dense(units=16, input_shape=(4,), activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=len(self.actions))
        ])
        self.model.summary()
        self.model.compile(SGD(learning_rate=0.2), "mse")
        # self.model.load_weights("model.h5")
        self.exp_replay = ExperienceReplay(max_memory=1000)

        # timer = Thread(target=self.action_timer)
        # timer.start()

    def reinforced_action_policy(self):
        pass
        # In [Ball-Y-Dist, ball-x-dist, wall_paddle_dist, wall_ball_dist)
        # Expand x2 -> 8
        # Expand x3 -> 24
        # Condense /8 -> 3
        # OUT [left, 0, right]

    def action_policy(self):
        step_size = 1
        # self.acceleration = random.choice([-1, 0, 1])
        # self.acceleration = -self.ball_distance/10
        self.velocity += step_size*self.acceleration

    def action_timer(self):
        while True:
            self.calculate_score()
            if self.goFlag:
                self.reinforced_action_policy()
                self.goFlag = False
            else:
                sleep(0.25)
                self.goFlag = True

    def calculate_score(self):
        hits = self.hits
        run_time = time() - self.start_time
        dist_bias = - abs((self.ball_distance_x / 400) ** 2)
        score = hits ** 2 + run_time ** 2 + dist_bias
        # score = hits ** 2 + run_time ** 2
        # score = hits
        # score = dist_bias
        self.score = score
        return score

    def copy(self):
        return self


class ExperienceReplay(object):
    def __init__(self, max_memory=500, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=50):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i + 1] = state_t

            # Compute Discounted Future Reward only for this action.
            # Don't compute it for other actions.
            # This way error for other actions will be zero and weights will not be adjusted for them.
            # Network will only learn for this action.
            targets[i] = model.predict(state_t)[0]

            # Predict reward for next state for this action.
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets