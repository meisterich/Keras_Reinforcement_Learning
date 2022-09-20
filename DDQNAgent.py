# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 19:18:22 2022

@author: Ue
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1:cpu, 0:first gpu; cpu faster on laptop?

import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import pickle
import math
import json
import tensorflow as tf
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() # usually using this for fastest performance
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler,PopulationBasedTraining
# from ray.tune.schedulers.pb2 import PB2
# from ray.tune import Trainable
# from ray.tune.integration.wandb import (WandbLoggerCallback,WandbTrainableMixin,wandb_mixin,)
#import wandb
# from wandb.keras import WandbCallback
# WANDB_API_KEY='fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1'



class ReplayMemory():
    def __init__(self, maxMemory = 20_000):
         self.buffer = []
         self.bufferMaxLength=maxMemory

    def addMemory(self,state,action,reward,nextstate,done):
         #print('adding memory')
         newMemory = (state, action, reward, nextstate,done)
         #print(newMemory)
         # if maxMemory size erreicht, entferne ältesten Eintrag
         if len(self.buffer) >= self.bufferMaxLength:
             self.buffer.pop(0)

         self.buffer.append( newMemory )
         return True

    def getBatch(self, size=None):
        #print('get batch with size %s'%size)
        if size == None or size > len(self.buffer):
            size = len(self.buffer)
        idx = np.random.choice(np.arange(len(self.buffer)), size=size, replace=False)
        batch = [self.buffer[i] for i in idx]
        #print(batch)
        return batch

    def getMemory(self):
        memory = self.buffer
        return memory

    def getSize(self):
        return len(self.buffer)

    def saveMemory(self,name):
        filename = name+".mem"
        dbfile = open(filename, 'wb')
        pickle.dump(self.buffer, dbfile)
        dbfile.close()
    def loadMemory(self,name):
        filename = name+".mem"
        dbfile = open(filename, 'rb')
        self.buffer = pickle.load(dbfile)
        dbfile.close()

#implements double deep Q-learning nach van hasselt et al. (2016)
class DDQAgent():
    def __init__(self, config,env=None):
        self.config = config
        self.discount=config['discount']
        self.lr=config['lr']
        if env:
            self.env = env
        else:
            self.env= gym.make(config['env'])
        self.actions=self.env.action_space.n
        print('action space: ',self.actions)
        self.observation_space=self.env.observation_space.shape[0]
        print('observation space: ',self.observation_space)
        epsConfig=config['epsilon']
        self.epsFunction =self.initEpsilon(epsConfig)
        self.epsMin=epsConfig[1]
        self.epsilon=self.setEpsilon(0)
        self.layerSizes = [config['layer1'],config['layer2'],config['layer3']]

        self.model = self.buildModel(self.actions,self.layerSizes)
        self.target_model = self.buildModel(self.actions,self.layerSizes)
        self.target_model.set_weights(self.model.get_weights())
        self.model_update_epochs=config['model_update_epochs']
        self.model_update_counter=0
        self.memory = ReplayMemory()
        self.df = pd.DataFrame()

    def buildModel(self,numActions,layerSizes):
        inputShape=[self.observation_space]
        model = Sequential()
        model.add(Dense(layerSizes[0], input_shape=inputShape,activation='tanh', kernel_initializer='random_normal'))
        model.add(Dense(layerSizes[1],activation='tanh', kernel_initializer='random_normal'))
        if layerSizes[2]:
                model.add(Dense(layerSizes[2],activation='tanh', kernel_initializer='random_normal'))
        model.add(Dense(numActions,activation='linear'))
        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
        model.summary()
        time.sleep(3)
        return model

    def updateMemory(self,state,action,reward, nextstate,done):
        self.memory.addMemory(state, action, reward, nextstate,done)

    def getAction(self,state):
        state = np.array(state)
        rand =np.random.rand()
        if rand > self.epsilon:
            state = state.reshape(1,-1)
            #faster:
            if tf.executing_eagerly():
                Qvalues = self.model(state)#.predict(state) for graph execution. model(state) faster in eager execution
            else:
                Qvalues = self.model.predict(state,verbose=0)

            action = np.argmax(Qvalues)
        else:
            action = np.random.choice(self.actions)

        return action

    def train(self,batchSize=512,epochs=32,verbose=2):
        
        minMemorySize = 10_000
        if self.memory.getSize() < minMemorySize:
            #print('not enough memory entrys to train')
            return
        print('training...')
        
        batches = self.memory.getBatch(batchSize*epochs)
        states = [i[0] for i in batches]
        next_states = [i[3] for i in batches]
        states = np.asarray(states)
        states.reshape(states.shape + (1,))
        next_states = np.asarray(next_states)
        next_states.reshape(next_states.shape + (1,))
        Qvalues = self.model.predict(states)#predict on whole batch-faster
        futureQvalues = self.model.predict(next_states)
        targetQvalues = self.target_model.predict(next_states)

        X=[]
        Y=[]
        #nach van Hasselt 2016:
        for i, (state,action,reward,next_state,done) in enumerate(batches):
            state=np.array(state)
            target = reward
            if not done:
                target = reward + self.discount * targetQvalues[i][np.argmax(futureQvalues[i])]#target/future andersherum?-no,richtig so
            Q = Qvalues[i]
            Q[action] = target
            X.append(state)
            Y.append(Q)
        X=np.array(X)
        Y=np.array(Y)

        cb= tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.model.fit(X,Y,epochs=10,batch_size=batchSize,callbacks=[cb],shuffle=False,verbose=0)

        self.model_update_counter+=1
        if self.model_update_counter > self.model_update_epochs:
            print('updating target model')
            old_weights = self.target_model.get_weights()
            new_weights = self.model.get_weights()
            self.target_model.set_weights([self.config['tau']*new_weights[i] + (1-self.config['tau'])*old_weights[i] for i in range(len(old_weights))])
            self.model_update_counter=0

    def initEpsilon(self,config):
        eps_max=config[0]
        eps_min=config[1]
        episodes=config[2]
        a=0.4#streckung
        b=4#steigung
        c=0.1#je kleiner desto später epsilon abgebaut
        episodes = int(episodes)
        x1 = np.linspace(0,episodes,num=episodes)
        x=(x1)/(a*episodes)
        y1 = -np.power(x,b)*c
        y2 = np.exp(y1)
        y= ((eps_max-eps_min) * (y2- np.min(y2)))/ (np.max(y2) - np.min(y2))+eps_min
        plt.figure()
        plt.plot(x1,y)
        plt.grid()
        plt.title('Epsilon Function')
        plt.show()
        return y

    def setEpsilon(self,episode):
        if episode < len(self.epsFunction):
            self.epsilon = self.epsFunction[episode]
        else:
            self.epsilon = self.epsMin


    def plot(self,df,save=False):
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.set_ylabel('Steps',)
        ax.yaxis.label.set_color('y')
        ax2 = ax.twinx()
        ax2.set_ylabel('Reward')
        ax2.yaxis.label.set_color('c')
        # ax.grid(which='minor',axis='y')
        ax3 =ax.twinx()
        ax3.set_ylabel('Epsilon')
        ax3.set_ylim([0, 1])
        ax3.yaxis.label.set_color('green')
        ax3.spines['right'].set_position(('outward', 60))

        df.plot(y=['epsilon'],ax=ax3,color='green',legend=False)
        df.plot(y=['score','score_avg'],ax=ax2,color=['c','blue'],legend=False)
        df.plot(y=['steps'],color='y',ax=ax,legend=False)

        #fig.suptitle(' lr: %3s, discount: %5s'%(lr,discount))
        title = str(self.__class__.__name__)+' '+str(self.config['env'])
        title += ('\n lr=%s, discount=%s,' %(self.config['lr'],self.config['discount']))
        layerSizes=self.layerSizes
        title += (' Net Layout %s'%layerSizes)
        fig.suptitle(title)
        handles1, _ = ax.get_legend_handles_labels()
        handles2, _ = ax2.get_legend_handles_labels()
        handles3, _ = ax3.get_legend_handles_labels()
        handles = handles1 + handles2 +handles3
        labels=[l.get_label() for l in handles]
        plt.legend(handles, labels,loc=3)
        ax.set_xlabel('Episodes')
        ax.grid(visible=True, which='major',axis='both', color='#666666', linestyle='-', alpha=0.5)
        # Show the minor grid lines with very faint and almost transparent grey lines
        ax.minorticks_on()
        ax.grid(visible=True, which='minor',axis='both', color='#999999', linestyle='-', alpha=0.2)
        plt.tight_layout()
        plt.show()

    def logMetrics(self,episode,score,steps,checkpoints=False):
        self.df.loc[episode,'score']=score
        self.df.loc[episode,'steps']=steps
        self.df.loc[episode,'score_avg'] =self.df['score'].rolling(100,min_periods=1).mean().iloc[-1]
        self.df['score_mean'] =self.df['score'].mean()
        self.df.loc[episode,'steps_avg'] =self.df['steps'].rolling(100,min_periods=1).mean().iloc[-1]
        self.df.loc[episode,'epsilon'] = self.epsilon
        score_avg = self.df.loc[episode,'score_mean']

        # if tune.is_session_enabled():
        #         tune.report(steps=steps,
        #                     score=score,
        #                     score_avg=self.df.loc[episode,'score_avg'],
        #                     episode=episode)	#log to ray tune	

        # if wandb.run:
        #     wandb.log({'steps':steps,
        #                 'score':score,
        #                 'score_avg':score_avg,
        #                 'epsilon':self.epsilon,
        #                 'episode':episode})
        #     #if best avg performance save model
        #     if self.df.loc[episode,'score_avg'] >= self.df.loc[episode,'score_avg'].max():
        #         print('Saving Model')
        #         self.model.save(os.path.join(wandb.run.dir, "bestModel.h5"))
        #         wandb.save(os.path.join(wandb.run.dir, "bestModel.h5"))

        if episode %50 ==0:# and (not tune.run):
                self.plot(self.df)

        #log checkpoints for PBT
        # if checkpoints:
        #     if episode%10 ==0:
        #         with tune.checkpoint_dir(step=episode) as checkpoint_dir:
        #             path = os.path.join(checkpoint_dir, "checkpoint")
        #             with open(path, "w") as f:
        #                 f.write(json.dumps({"episode": episode}))
        #             #print('Saving checkpoint to: ', checkpoint_dir)
        #             path = os.path.join(checkpoint_dir, "Model.h5")
        #             #print('Saving model to: ', path)
        #             self.model.save_model(path)

    def run(self, checkpoints = False,start=0):
        for i in range(start,config['episodes']):#,desc ='Episodes progress',unit='episode'):

            self.setEpsilon(i)
            state=self.env.reset()
            done=False
            steps =0
            score=0
            render = False
            if i %10 ==0:# and not tune.run:
                render = False
            while(not done):
                action = self.getAction(state)
                #print(action)
                new_state, reward, done, _= self.env.step(action)
                score += reward
                self.updateMemory(state, action, reward, new_state,done)
                state = new_state
                steps+=1
                if render:
                    self.env.render()

                if done:
                    self.logMetrics(i,score,steps,checkpoints)
                    print("episode: {}/{}, score: {:.3f}, average: {:.2f}".format(i, config['episodes'], score, self.df['score_avg'].iloc[-1]))
                    self.train(batchSize=config['batchSize'],verbose =0)
                    
        self.env.close()
        #wandb.finish()



    def testAgent(self):
        path =os.path.join(os.getcwd(),'Models','DDQN','LL','bestModel.h5')
        try:
            self.model = load_model(path)
        except:
            print('Model not found at: ',path)
            return
        print('loaded Model from: ',path)
        scores = []
        for i in range(10):
            state = self.env.reset()
            done = False
            score = 0
            self.epsilon=0
            while not done:
                action = self.getAction(state)
                new_state, reward, done, _ = self.env.step(action)
                score += reward
                state = new_state
                self.env.render()
            print("episode: {}/{}, score: {:.3f}".format(i+1, 10, score))
            scores.append(score)
        print('Average score: ',np.mean(scores))
        self.env.close()


if __name__ == '__main__':
    #default config
    config={'env':'LunarLander-v2','episodes': 10000,'discount':0.99,'lr':0.0001,'epsilon':[0.95,0.02,1000],
            'layer1':64,'layer2':64,'layer3':None,'model_update_epochs':5,'batchSize':512,'tau':0.2,
            'loadModel':False,'wandb':None}
    #wandb.init(config=config, entity="meisterich", project="AKI", group ='LunarLanderWind',mode='online',job_type='DDQN')
    # env = gym.make('LunarLander-v2',enable_wind=True)
    # agent = DDQAgent(config,env)
    #start normal training:
    #agent.run()

    #test Agent with best model:
    env = gym.make('LunarLander-v2',enable_wind=True)
    agent = DDQAgent(config,env)
    agent.testAgent()




















    #from here for hyperparameter tuning:
    # def startTraining(config, checkpoint_dir=None):
    #     wandb.init(config=config, entity="meisterich", project="AKI", group ='LunarLanderWind',mode='online',job_type='DDQN')
    #     env = gym.make('LunarLander-v2',enable_wind=True)
    #     agent = DDQAgent(config,env)
    #     start =1
    #     ckpts = False
    #     if checkpoint_dir:
    #         path = os.path.join(checkpoint_dir, "Model.h5")
    #         agent.model = load_model(path)
    #         with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
    #             state = json.loads(f.read())
    #             start = state["episode"] + 1
    #             ckpts = True
    #     agent.run(checkpoints=ckpts,start=start)
    #     agent.plot(agent.df)
    #     print('Done')
    #     return agent.df

    # def startHyperparameterSearch(config):
    #     config["wandb"]= {'project': 'AKI','api_key':'fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1',
    #                 'group':'LunarLanderDDQ_wind','mode':'online'}
    #     #Hyperparameter search:
    #     parameters = {'lr':tune.loguniform(0.0001, 0.001),
    #                 'batch_size':tune.choice([64,128,256,512,1024,2048]),'tau':tune.quniform(0.1,0.5,0.1),
    #                 'layer1':tune.quniform(32, 100,1),'layer2':tune.quniform(32, 100,1),'layer3':tune.choice([None,8,16]),
    #                 'model_update_epochs':tune.quniform(5, 500, 1)}
 
    #     config['lr'] = parameters['lr']
    #     config['batch_size'] = parameters['batch_size']
    #     config['tau'] = parameters['tau']
    #     config['model_update_epochs'] = parameters['model_update_epochs']

    #     # #doesnt work with pbt
    #     # config['layer1'] = parameters['layer1']
    #     # config['layer2'] = parameters['layer2']
    #     # config['layer3'] = parameters['layer3']

    #     pbt = PopulationBasedTraining(time_attr='episode',metric="score_avg",mode="max",perturbation_interval=100,burn_in_period=1000,
    #                                 hyperparam_mutations={'lr':parameters['lr'],'batch_size':parameters['batch_size'],
    #                                                     #'layer1':parameters['layer1'],'layer2':parameters['layer2'],'layer3':parameters['layer3'],
    #                                                     'model_update_epochs':parameters['model_update_epochs']},)

    #     pbt = PB2(time_attr='episode',metric="score_avg",mode="max",perturbation_interval=100,
    #                  hyperparam_bounds={"lr": [0.0001, 0.001],'batchsize':[32,1024],
    #                                     'model_update_epochs':[5,500],})

    #     asha =ASHAScheduler(metric="score_avg",mode="max",time_attr='episode',grace_period=1000,max_t=config['episodes'])

    #     # ray.init(logging_level=logging.FATAL)
    #     #starts hyperparameter search with
    #     analysis = tune.run(startTraining,
    #                         config=config,
    #                         num_samples=20,
    #                         scheduler=asha,
    #                         resources_per_trial={'gpu': 1},
    #                         progress_reporter=CLIReporter(max_report_frequency=30),
    #                         fail_fast=True,
    #                         local_dir="checkPoints/",
    #                         sync_config=tune.SyncConfig(),
    #                         keep_checkpoints_num=1,
    #                         checkpoint_score_attr='reward_avg',
    #                         #resume="AUTO",
    #                         )
    #     # ray.shutdown()
    #     print("Best hyperparameters found were: ", analysis.get_best_config("score_avg",'max'))
    #     dfs = analysis.trial_dataframes()
    #     ax = None  # This plots everything on the same plot
    #     for d in dfs.values():
    #         ax = d.plot('episodes','reward_avg',ax=ax)
    #     plt.ylabel('reward_avg')
    #     plt.xlabel('Episodes')

    # def startHyperparameterSearchWandb():
    #     sweep_config = {
    #         "name" : 'DDQAgent', "method" : "bayes",
    #         'metric': {'goal': 'minimize', 'name': 'reward_avg'},
    #         'early_terminate':{'type':'hyperband','min_iter': 3},
    #         'parameters':{
    #             'batch_size':{'values':[16,32,64,128,256,512]},
    #             'lr':{'distribution': 'log_uniform', 'min': math.log(0.0001),'max': math.log(0.1)},
    #             'layerSizes':{'values':[(32,16),(32,32),(64,32),(32,16,8)]},
    #             }}
    #     sweep_config['parameters'].update({
    #         'epoches':{'value':6000},'discount':{'value':0.99},'episodes':{'value':128},
    #         'epsilon':{'value':[0.99,0.1,3000]}
    #         })
    #     sweep_id = wandb.sweep(sweep_config, project='AKI')
    #     wandb.agent(sweep_id, startTraining,count=30)
    #     return

    # #start hyperparameter search:
    # startHyperparameterSearch(config)
  

