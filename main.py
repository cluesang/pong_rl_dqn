from game import Game
from agent import Agent
import sys, os
from datetime import datetime
import time
from loggers import saveDictionaryToCSV\
                    , openTrainingConfig\
                    , saveTrainingConfig\
                    , saveModelJsonSummary
from tensorflow.keras.models import load_model


def train(silent=False, sessionId=None):
    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M:%S:%p")
    isResumingSession = (sessionId != None)
    if(not isResumingSession):
        sessionId = str(int(time.time()))
        os.makedirs("./session_"+sessionId+"/")

    else:
        if not os.path.exists("./session_"+sessionId):
            print("Error: cannot find session folder for session id: "+sessionId)
            print("try restarting without providing a session id or placing the session"\
                " folder in the same directory as this script.")
            return

    sessionFolderpath = "./session_"+sessionId+"/"
    configFilename = sessionFolderpath+"config_"+sessionId+".json"
    historyFilename = sessionFolderpath+"history_"+sessionId+".csv"
    modelFilename = sessionFolderpath+"model"
    targetModelFilename = sessionFolderpath+"model_target"
    modelSummaryFilename = sessionFolderpath+"model_"+sessionId+".json"
    targetModelSummaryFilename = sessionFolderpath+"target_model_"+sessionId+".json"
    videoFolderPath = sessionFolderpath+"videos/"

    saved_model = None
    saved_model_target = None
    if(isResumingSession):    
        config = openTrainingConfig(configFilename)
        saved_model = load_model(modelFilename)
        saved_model_target = load_model(targetModelFilename)
    else:
        config = {
            'episode_number': 0
        ,   'datetime': timestamp
        ,   'sessionId': sessionId
        }
    
    agent = Agent(verbose=True
                ,save_directory=sessionFolderpath
                ,resume_model=saved_model
                ,resume_target_model=saved_model_target)
    
    # if first run of this model/session
    # then dump model summary and save initial empty models
    if(not isResumingSession):
        saveModelJsonSummary(agent.model.to_json(),modelSummaryFilename)
        saveModelJsonSummary(agent.model_target.to_json(),targetModelSummaryFilename)
        agent.saveModels()

    game = Game('PongDeterministic-v4'
                ,agent
                ,verbose=False
                ,filePath=videoFolderPath
                )
    max_score = -21
    i = int(config['episode_number'])
    recordEpisode = False
    while True:
        timesteps = agent.total_timesteps
        time_elapsed = time.time() # start timer
        
        # play the game
        score = game.play(showRender=True, recordRender=recordEpisode)
        
        # update max score
        if score > max_score:
            max_score = score

        episodeData = {
            'sessionId': sessionId
        ,   'episode_number': str(i)
        ,   'steps': str(agent.total_timesteps - timesteps)
        ,   'duration': str(time.time() - time_elapsed)
        ,   'score': str(score)
        ,   'max_score': str(max_score)
        ,   'epsilon': str(agent.epsilon)
        }
        print(episodeData)
        saveDictionaryToCSV(episodeData,historyFilename) # save history

        config['episode_number'] = str(i)
        saveTrainingConfig(config,configFilename) # saveConfig

        if i%100==0:
            recordEpisode = True
        i += 1

if __name__ == '__main__':
    try:
       #train()
       # to resume training an old session
       train(sessionId="1638589724")

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)