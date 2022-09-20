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
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Dense,Input, Activation, BatchNormalization, Dropout, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() # usually using this for fastest performance
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler,PopulationBasedTraining
# from ray.tune.schedulers.pb2 import PB2
# import wandb
# from wandb.keras import WandbCallback
# WANDB_API_KEY='fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1'




class ReplayMemory():

    def __init__(self, maxMemory = 50_000):
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

class NAFModel():
    def __init__(self,state_space,action_space,lr=0.0003,layers=[64,64]):
        state_input = Input(state_space)
        action_input = Input(shape=(action_space,))
        X = Dense(layers[0], activation='relu')(state_input)
        X = Dense(layers[1], activation='relu')(X)
        if len(layers)>2:
            X = Dense(layers[2], activation='relu')(X)
        V = Dense(1, activation='linear')(X)
        mu = Dense(action_space, activation='tanh')(X)
        L = Dense(action_space * (action_space+ 1)/2,activation='tanh')(X)
        action = action_input[:,:action_space]

        pivot = 0
        rows = []
        for idx in range(action_space):
            count = action_space - idx
            diag_elem = tf.exp(tf.slice(L, (0, pivot), (-1, 1)))
            non_diag_elems = tf.slice(L, (0, pivot + 1), (-1, count - 1))
            row = tf.pad(tensor=tf.concat((diag_elem, non_diag_elems), 1), paddings=((0, 0), (idx, 0)))
            rows.append(row)
            pivot += count
        L = tf.transpose(a=tf.stack(rows, axis=1), perm=(0, 2, 1))
        P = tf.matmul(L, tf.transpose(a=L, perm=(0, 2, 1)))
        tmp = tf.expand_dims(action - mu, -1)
        A = -tf.multiply(tf.matmul(tf.transpose(a=tmp, perm=[0, 2, 1]), tf.matmul(P, tmp)), tf.constant(0.5))
        A = tf.reshape(A, [-1, 1])
        Q = tf.add(A, V)

        self.QModel=Model(inputs=[state_input, action_input], outputs=Q)
        self.QModel.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        self.QModel.summary()
        self.ActionModel=Model(inputs=state_input, outputs=mu)
        self.ValueModel=Model(inputs=state_input, outputs=V)

    def getAction(self,state):
        state=np.array(state)
        state=np.expand_dims(state,axis=0)
        if tf.executing_eagerly():
            action = self.ActionModel(state)
        else:
            action = self.ActionModel.predict(state,verbose=0)
        return action

    def getValue(self,state):
        state=np.array(state)
        if tf.executing_eagerly():
            value = self.ValueModel(state)
        else:
            value = self.ValueModel.predict(state,verbose=0)
        return value

    def train_model(self,states,actions,target,batchSize):
        states=np.array(states)
        actions=np.array(actions)
        # states = tf.convert_to_tensor(states, dtype=tf.float32)
        # actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        #target = tf.convert_to_tensor(target, dtype=tf.float32)
        cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.QModel.fit([states,actions],target,batch_size=batchSize,epochs=20,callbacks=[cb],verbose=0)


#implements double deep Q-learning nach van hasselt et al. (2016)
class DDQNAFAgent():
    def __init__(self, config,env):
        self.config = config
        self.discount=config['discount']
        self.lr=config['lr']
        self.env = env

        self.action_space = self.env.action_space.shape[0]
        self.observation_space = self.env.observation_space.shape
        print('action space: ',self.action_space)
        print('observation space: ',self.observation_space)
        print('running eagerly: ',tf.executing_eagerly())
        epsConfig=config['epsilon']
        self.epsFunction =self.initEpsilon(epsConfig)
        self.epsMin=epsConfig[1]
        self.epsilon=self.setEpsilon(0)
        #self.layerSizes = [config['layer1'],config['layer2'],config['layer3']]

        self.model = NAFModel(self.observation_space,self.action_space,lr=self.lr, layers=config['layers'])
        self.target_model = NAFModel(self.observation_space,self.action_space,lr=self.lr, layers=config['layers'])
        self.target_model.QModel.set_weights(self.model.QModel.get_weights())
        self.model_update_epochs=config['model_update_epochs']
        self.model_update_counter=0
        self.memory = ReplayMemory()
        self.df = pd.DataFrame()
        if config['load']:
            path =os.path.join(os.getcwd(),'Models','DQNAF','BW_h','bestModel_weights.h5')
            self.model.QModel.load_weights(path)
            print("Actor Model Loaded from: ",path)


    def updateMemory(self,state,action,reward, nextstate,done):
        self.memory.addMemory(state, action, reward, nextstate,done)

    def getAction(self,state):
        state = np.array(state)
        #state = np.expand_dims(state, axis=0)
        mu = self.model.getAction(state)
        action = mu + self.epsilon * np.random.randn(self.action_space)
        return action[0]

    def train(self,batchSize=128,epochs=256):
        
        minMemorySize = 5000
        if self.memory.getSize() < minMemorySize:
            print('not enough memory entrys to train')
            return
        print('training...')
        
        batch = self.memory.getBatch(batchSize*epochs)#get whole menmory
        states = [i[0] for i in batch]
        actions = [i[1] for i in batch]
        rewards = [i[2] for i in batch]
        next_states = [i[3] for i in batch]
        dones = [i[4] for i in batch]
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        states.reshape(states.shape + (1,))
        next_states.reshape(next_states.shape + (1,))

        values = self.target_model.getValue(next_states)
        values = np.squeeze(values)

        target = rewards + (1-dones)* self.discount * values

        self.model.train_model(states,actions,target,batchSize)
        
        self.model_update_counter+=1
        if self.model_update_counter > self.model_update_epochs:
            print('updating target model')
            old_weights = self.target_model.QModel.get_weights()
            new_weights = self.model.QModel.get_weights()
            self.target_model.QModel.set_weights([self.config['tau']*new_weights[i] + (1-self.config['tau'])*old_weights[i] for i in range(len(old_weights))])
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
        self.df.loc[episode,'score_avg100'] =self.df['score'].rolling(100,min_periods=1).mean().iloc[-1]
        self.df.loc[episode,'score_avg'] =self.df['score'].mean()
        self.df.loc[episode,'steps_avg'] =self.df['steps'].rolling(100,min_periods=1).mean().iloc[-1]
        self.df.loc[episode,'epsilon'] = self.epsilon

        # if tune.is_session_enabled():
        #         tune.report(steps=steps,
        #                     score=score,
        #                     score_avg=self.df.loc[episode,'score_avg'],
        #                     episode=episode)	#log to ray tune	

        # if wandb.run:
        #     wandb.log({'steps':steps,
        #                 'score':score,
        #                 'score_avg':self.df.loc[episode,'score_avg'],
        #                 'score_avg100':self.df.loc[episode,'score_avg100'],
        #                 'epsilon':self.epsilon,
        #                 'episode':episode})
        #     #if best avg performance save model
        #     if self.df.loc[episode,'score_avg100'] >= self.df['score_avg100'].max():
        #         print('Saving Model')
        #         self.model.QModel.save_weights(os.path.join(wandb.run.dir, "bestModel_weights.h5"))
        #         wandb.save(os.path.join(wandb.run.dir, "bestModel_weights.h5"))

        # if episode %50 ==0:# and (not tune.run):
        #         #self.plot(self.df)
        #         pass

        # #log checkpoints for PBT
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

    def run(self):
        for i in range(config['episodes']):#,desc ='Episodes progress',unit='episode'):
            print('\nEpisode %s/%s'%(i,config['episodes']))

            self.setEpsilon(i)
            state=self.env.reset()
            done=False
            steps =0
            score=0
            render = False
            if i %2 ==0:# and not tune.run:
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
                    self.logMetrics(i,score,steps)
                    print(' - steps: %s, score %s, average: %s '%(steps,score,self.df['score_avg'].iloc[-1]))
                    self.train(batchSize=config['batchSize'])
                    
        self.env.close()

    
    
    def testAgent(self,env='LL'):
        print('Testing Agent')
        path =os.path.join(os.getcwd(),'Models','DQNAF',env,'bestModel_weights.h5')
        try:
            self.model.QModel.load_weights(path)
        except:
            print('Model not found at: ',path)
            return
        print('loaded Model from: ',path)
        time.sleep(1)
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
    config={'episodes': 20000,'discount':0.99,'lr':0.0001,'epsilon':[0.9,0.1,9000],
            'layers':[64,64],'model_update_epochs':10,'batchSize':256,'tau':0.9,
            'load':False}

    #start normal training:
    # wandb.init(config=config, entity="meisterich", project="AKI", group ='BipedalWalker_hardcore',mode='online',job_type='DQNAF_pretrained')
    # env = gym.make('BipedalWalker-v3',hardcore=True)
    # agent = DDQNAFAgent(config,env)
    # agent.run()


#test Environments with best model
def test():
    #wait for keyboard input
    inp= input('which environment to test?\n type\n LL: LunarLander;\n BW: BipedalWalker;\n BW_h: BipedalWalker_Hardcore')
    if inp == 'LL':
        env = gym.make('LunarLanderContinuous-v2',enable_wind=True)
        print('start testing LunarLander') 
    elif inp == 'BW':
        env = gym.make('BipedalWalker-v3',hardcore=False)
        print('start testing BipedalWalker')
    elif inp== 'BW_h':
        env = gym.make('BipedalWalker-v3',hardcore=True)
        print('start testing BipedalWalker_Hardcore')
    time.sleep(1)
    agent = DDQNAFAgent(config,env)
    agent.testAgent(env=inp)

test()














    # # # #from here for hyperparameter tuning:
    # def startTraining(config=None):
    #     wandb.login(key='fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1')
    #     wandb.init(config=config, entity="meisterich", project="AKI", group ='BipedalWalker_hardcore',mode='online',job_type='DQNAF_pretrained')
    #     config = wandb.config
    #     env = gym.make('BipedalWalker-v3',hardcore=True)
    #     agent = DDQNAFAgent(config,env)
    #     agent.run()
    #     agent.plot(agent.df)
    #     print('Done')
    #     return agent.df

    # def startHyperparameterSearch(config):
    #     config["wandb"]= {'project': 'AKI','api_key':'fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1',
    #                 'group':'LunarLanderDDQ_wind','mode':'online'}
    #     #Hyperparameter search:
    #     parameters = {'lr':tune.loguniform(0.0001, 0.001),
    #                 'batch_size':tune.choice([32,64,128,256,512]),
    #                 'layer1':tune.quniform(32, 100,1),'layer2':tune.quniform(32, 100,1),'layer3':tune.choice([None,8,16]),
    #                 'model_update_epochs':tune.quniform(5, 500, 1),'tau':tune.uniform(0.1, 0.9),
    #                 'eps_min':tune.quniform(0.1, 0.9,0.1)}
 
    #     config['lr'] = parameters['lr']
    #     config['batch_size'] = parameters['batch_size']
    #     config['model_update_epochs'] = parameters['model_update_epochs']
    #     config['tau'] = parameters['tau']

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

    #     asha =ASHAScheduler(metric="reward_avg",mode="max",time_attr='episode',grace_period=1000,max_t=config['episodes'])

    #     # ray.init(logging_level=logging.FATAL)
    #     #starts hyperparameter search with
    #     analysis = tune.run(startTraining,
    #                         config=config,
    #                         num_samples=5,
    #                         scheduler=pbt,
    #                         resources_per_trial={'cpu': 1},
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


    # def startHyperparameterSearchWandb(config):
    #     sweep_config = {
    #         "name" : 'BWDQNAF', "method" : "bayes",
    #         'metric': {'goal': 'maximize', 'name': 'score_avg100'},
    #         'early_terminate':{'type':'hyperband','max_iter': 10_000,'s':1},
    #         'parameters':{
    #             'batch_size':{'distribution': 'q_uniform', 'min': 512, 'max': 2048, 'q': 256},
    #             'lr':{'distribution': 'q_log_uniform', 'min': math.log(0.00005),'max': math.log(0.0002),'q': 0.00001},
    #             'tau':{'distribution': 'q_uniform', 'min': 0.5, 'max': 0.9, 'q': 0.1},
    #             'model_update_epochs':{'distribution': 'q_uniform', 'min': 1,'max': 50,'q':1},
    #             }}
    #     sweep_config['parameters'].update({
    #         'episodes': {'value': config['episodes']},
    #         'discount': {'value': config['discount']},
    #         'epsilon': {'value': config['epsilon']},
    #     })
    #     sweep_id = wandb.sweep(sweep_config, project='AKI')
    #     wandb.agent(sweep_id, startTraining,count=30)
    #     return

    # #start hyperparameter search:
    # startHyperparameterSearchWandb(config)
  
