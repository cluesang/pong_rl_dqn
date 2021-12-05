import gym
import numpy as np
import os
import sys
import random
import cv2

def resize_frame(frame):
    frame = frame[30:-12,5:-4]
    frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame

class Game():
    def __init__(self, game_name, agent=None, verbose=False, filePath="./video_capture/", ):
        self.name = game_name
        self.env = gym.make(self.name, render_mode='rgb_array')
        self.env.reset()
        self.verbose = verbose
        self.filePath = filePath
        self.filterdEpisodeFrames = []
        self.agent = agent
        self.scoreTotal = 0

        starting_frame = resize_frame(self.env.step(0)[0])

        dummy_action = 0
        dummy_reward = 0
        dummy_done = False
        for i in range(3):
            agent.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)

        pass

    def reset(self):
        self.filterdEpisodeFrames = []
        self.scoreTotal = 0
        self.env.reset()    

    def step(self,showRender=False):
        
        # use the agents last action to feed to game step
        frame, reward, done, info = self.env.step(self.agent.get_last_action())
        frame = resize_frame(frame)
        # so we can save a filtered frame video
        self.filterdEpisodeFrames.append(frame)
        # the * operatore unpacks the list provided by the function call
        new_frame_set = [*self.agent.get_last_n_frames(3),frame]
        new_frame_set = np.moveaxis(new_frame_set,0,2)/255 #We have to do this to get it into keras's goofy format of [batch_size,rows,columns,channels]
        new_frame_set = np.expand_dims(new_frame_set,0) #^^^

        if self.agent==None:
            next_action = random.sample([0,2,5],1)[0]
        else:
            next_action = self.agent.get_next_action(new_frame_set)
        

        if self.agent != None:
            self.agent.add_experience(frame
                                    ,reward
                                    ,next_action
                                    ,done)

        # if done:
        #     return (self.scoreTotal + reward), True

        if self.verbose:
            print("action: {}\tobvservation: {}\treward: {}\tdone: {}\tinfo: {}"\
                .format(action, observation, reward, done, info))
            # print("done: {}"\
            #     .format(done))
        if showRender:
            self.env.render()

        # #9: If the threshold memory is satisfied, make the agent learn from memory
        # if len(self.agent.memory.frames) > self.agent.starting_mem_len:
        #     # print("learn from membor")
        #     self.agent.learn(debug)

        return reward, done

    def play(self,showRender=False,recordRender=False,recordFilteredRender=False):
        if recordRender:
            self.env = gym.wrappers.Monitor(self.env,self.filePath,force=True)
        self.reset()
        
        while True:
            score,done = self.step(showRender)
            self.scoreTotal += score
            if done:
                if recordFilteredRender:
                    out = cv2.VideoWriter(self.filePath+'episode.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84), False)
                    for frame in self.filterdEpisodeFrames:
                        out.write(frame)
                    out.release()
                break
        return self.scoreTotal


if __name__ == '__main__':
    try:
       game = Game('PongDeterministic-v4',verbose=True,filePath="./video_capture/")
    #    game.step(showRender=True)
       game.play(showRender=True,recordRender=False, recordFilteredRender=False)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)