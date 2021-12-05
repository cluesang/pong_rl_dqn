# PONG RL DQN
## Team: Positive Reinforcement
#### members:
- Aaron Patterson (ampatt11@asu.edu)
- Christopher D. Lue Sang (cdluesan@asu.edu)
- David Pierpont (dvpierpo@asu.edu)
- Zhaonian Wang (zwang709@asu.edu)

Dependencies:

Open AI's Gym library.
See this getting started: https://gym.openai.com/docs/

Typically:
    
    pip install gym
    pip install gym["Pong-v0"]

You may have trouble getting the emulator running on a windows machine. You mileage may vary with the above process.

General Usage:

    python main.py

## Description
This progam implements a deep Q reinforcement learning strategy on the the game pong. 
Running main will produce a 'training session'. The current unix timestamp will be the session's id. A folder titled 'session_\<sessionid\>' will be created in the run directory. 

The session folder will be populated with the tensorflow model summary (.json), as well as the episodic training data (.csv), config filea and history data. A subfolder named "model" contains the full tensorflow/keras model (.pb files, assets, variables)

    ./config_<session_id>.json
    ./history_<session_id>.csv
    ./model_<session_id>.json (model summary info)
    ./target_model_<session_id>.json (target model summary info. used in implementation logic)
    ./model/assets/ (typically blank)
    ./model/varialbes/
    ./model/keras_metadata.pb
    ./model/saved_model.pb
## Within main.py
You can resume a training session by providing a session id as a sting to the train function:

    train(sessionId="1638587126")
If the appropriate files exist, training will resume.

### Code/Repo Maintainer: cdluesan@asu.edu

## References
This code was heavily inspired by this write up: 
https://towardsdatascience.com/getting-an-ai-to-play-atari-pong-with-deep-reinforcement-learning-47b0c56e78ae
