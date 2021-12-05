import numpy as np
import enum
import random
from memory import Memory
import os
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import sys

class PongActions(enum.Enum):
    do_nothing = 0
    up = 2
    down = 5

class Agent():
    def __init__(self
                ,max_mem_len=750000
                ,save_directory="./session/"
                ,resume_model=None
                ,resume_target_model=None
                ,starting_epsilon=1
                ,learn_rate=0.00025
                ,verbose=False):

        self.max_mem_len = max_mem_len
        self.memory = Memory(max_mem_len)
        self.save_directory = save_directory
        self.epsilon = starting_epsilon
        self.epsilon_decay = .9/100000
        self.epsilon_min = .05
        self.learn_rate = learn_rate
        self.verbose = verbose
        self.learns = 0
        self.gamma = .95
        self.start_learning_count = 50000
        self.total_timesteps = 0
        self.possible_actions = [act.value for act in PongActions]
        if (resume_model == None):
            self.model = self.build_model()
            self.model_target = clone_model(self.model)
        else:
            self.model = resume_model
            self.model_target = resume_target_model
        
    def build_model(self):
        model = Sequential()
        model.add(Input((84,84,4)))
        model.add(Conv2D(filters = 32\
                        ,kernel_size = (8,8)\
                        ,strides = 4\
                        ,data_format="channels_last"\
                        ,activation = 'relu'\
                        ,kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)\
                        )\
                )
        model.add(Conv2D(filters = 64\
                        ,kernel_size = (4,4)\
                        ,strides = 2\
                        ,data_format="channels_last"\
                        ,activation = 'relu'\
                        ,kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)\
                        )\
                    )
        model.add(Conv2D(filters = 64\
                        ,kernel_size = (3,3)\
                        ,strides = 1\
                        ,data_format="channels_last"\
                        ,activation = 'relu'\
                        ,kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)\
                        )\
                    )
        model.add(Flatten())
        model.add(Dense(512,activation = 'relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(len(self.possible_actions), activation = 'linear'))
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        if self.verbose:
            model.summary()
            print('\nAgent Initialized\n')
        return model

    def add_experience(self, observation, reward, action, done):
        self.memory.add_experience(observation, reward, action, done)
        if not done:
            if len(self.memory.frames) > self.start_learning_count:
                self.learn()
    
    def get_last_n_frames(self,n):
        frames = []
        for i in range(-1,-1*(n+1),-1):
            frames.append(self.memory.frames[i])
        return frames

    def get_last_action(self):
        return self.memory.actions[-1]
    
    def saveModels(self):
        self.model.save(self.save_directory+"model")
        print("saving model")
        self.model_target.save(self.save_directory+"model_target")
        print("saving model target")

    def get_next_action(self,game_frame_set=None): 
        self.total_timesteps += 1
        if self.total_timesteps % 50000 == 0:
            self.saveModels()
        # get random/explore action if we're just starting 
        # to learn
        if np.random.rand() < self.epsilon:            
           return random.sample([act.value for act in PongActions],1)[0]

        """ Otherwise Use our model to pull the pull the best action we can"""
        a_index = np.argmax(self.model.predict(game_frame_set))
        return self.possible_actions[a_index]
    
    def validate_memory_set(self,index):
        if self.memory.done_flags[index-3] or self.memory.done_flags[index-2] or self.memory.done_flags[index-1] or self.memory.done_flags[index]:
            return False
        else:
            return True
    
    def seed_game_states_from_memory(self):
        """First we need 32 random valid indicies"""
        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        while len(states) < 32:
            index = np.random.randint(4,len(self.memory.frames) - 1)
            if self.validate_memory_set(index):
                state = [self.memory.frames[index-3]\
                        , self.memory.frames[index-2]\
                        , self.memory.frames[index-1]\
                        , self.memory.frames[index]]
                state = np.moveaxis(state,0,2)/255
                next_state = [self.memory.frames[index-2]\
                            , self.memory.frames[index-1]\
                            , self.memory.frames[index]\
                            , self.memory.frames[index+1]]
                next_state = np.moveaxis(next_state,0,2)/255

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index+1])
                next_done_flags.append(self.memory.done_flags[index+1])

        return states, next_states, actions_taken, next_rewards, next_done_flags

    def associate_state_and_Q_values(self, labels, actions_taken, next_rewards, next_state_values, next_done_flags):
        for i in range(32):
            action = self.possible_actions.index(actions_taken[i])
            labels[i][action] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * max(next_state_values[i])
            return labels

    def learn(self):
        ## seed state
        states, next_states, \
        actions_taken, next_rewards, \
        next_done_flags = self.seed_game_states_from_memory()

        ## predict labels aka rewards and q value##
        labels = self.model.predict(np.array(states))

        ## predict next state ##
        next_state_values = self.model_target.predict(np.array(next_states))

        ## associate game action with rewards and q value ##
        labels = self.associate_state_and_Q_values(labels, actions_taken, next_rewards, next_state_values, next_done_flags)

        ## train model with state and labels/rewards
        self.model.fit(np.array(states),labels,batch_size = 32, epochs = 1, verbose = 0)

        ## decrease episillon so we take less and less random actions ##
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1

        ## periodically save our model ##
        if self.learns % 10000 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')


if __name__ == '__main__':
    try:
      pass
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)