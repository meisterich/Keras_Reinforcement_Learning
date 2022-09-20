import os

from pandas import value_counts
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1:cpu, 0:first gpu
import random
import gym
print('gym version:', gym.__version__)
import pylab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import tensorflow as tf
print('tf version ',tf.__version__)
from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import copy

# import ray
# from ray import tune
# print('rayversion:', ray.__version__)
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler,PopulationBasedTraining
# from ray.tune.schedulers.pb2 import PB2
# from ray.tune.integration.wandb import WandbLoggerCallback,wandb_mixin
# from ray.tune.logger import DEFAULT_LOGGERS
#import wandb
# from wandb.keras import WandbCallback
# WANDB_API_KEY='fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1'


class PGAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, config,env):
        # Initialization
        # Environment and PPO parameters
        self.config = config
        if env:
            self.env = env
            print('env set')
        else:
            self.env_name = config['env']
            self.env = gym.make(self.env_name)
            print('making env', self.env_name)
        #self.env = gym.make(self.env_name)#,enable_wind=True)
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape
        print('action space', self.action_space)
        print('state space', self.state_space)
        self.EPISODES = config['episodes'] # total episodes to train through all environments
        self.lr = config['lr']
        self.alpha = config['alpha']
        self.trainingBatchSize = config['batchsize']
        self.batch_full = False
        self.state_memory = []
        self.action_memory = []
        self.action_probs_memory = []
        self.reward_memory = []
        self.done_memory = []
        self.next_state_memory = []
        self.df = pd.DataFrame()


        self.layerSizes = [config['layer1'],config['layer2'],config['layer3']]
        self.optimizer = Adam

        self.replay_count = 0

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        #self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        #self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Actor = self.build_model(self.layerSizes, lr=self.lr, optimizer = self.optimizer,af = config['activation'])



    def build_model(self, layerSizes, lr, optimizer, af='tanh'):
        X_input = Input(self.state_space)

        X = Dense(layerSizes[0], activation=af, kernel_initializer='random_uniform')(X_input)
        X = Dense(layerSizes[1], activation=af, kernel_initializer='random_uniform')(X)
        if layerSizes[2]:
            X = Dense(layerSizes[2], activation=af, kernel_initializer=tf.random_normal_initializer(stddev=0.5))(X)
        #X = Dense(64, activation="elu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.categorical_crossentropy_custom, optimizer=optimizer(lr=lr))
        self.Actor.summary()


        return self.Actor

    def categorical_crossentropy_custom(self, y_true, y_pred):

        loss = K.categorical_crossentropy(y_true,y_pred,from_logits=False)

        #entropy gets agent to explore more
        ENTROPY = self.config['entropy']#standard =0.01
        entropy_loss = (y_pred * K.log(y_pred + 1e-10))
        entropy_loss = -ENTROPY * K.mean(entropy_loss)

        return loss - entropy_loss



    def act(self, state,test=False):
        probs = self.Actor(state)[0].numpy().astype('float64')
        probs[np.isnan(probs)] = 0.1
        probs = probs/np.sum(probs)
        #probs[-1] = 1 - np.sum(probs[0:-1])
        # if np.sum(probs) != 1.0:
        #     print('probs: ', np.sum(probs))
        if test:
            action = np.argmax(probs)
            return action
        action = np.random.choice(self.action_space, p=probs)
        return action, probs



    def train(self, states, actions, action_probs, rewards, next_states, dones):
        print("Training...")
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        #print(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        action_probs = np.vstack(action_probs)
        probs_centered = actions - action_probs#centered around 0, for pushing actions not taken higher when bad reward and actions taken lower


        # Compute discounted rewards and advantages
        returns = self.calculateReturns(rewards)
        returns =np.vstack(returns)

        gradient = probs_centered * returns
        Y= action_probs + self.alpha*gradient

        self.Actor.fit(states, Y, epochs =1, verbose=0, shuffle=True)


    #Return : discounted kumulative reward G_t aka rewards to go
    def calculateReturns(self,rewards,discount = 0.99):
        '''
        calculate discounted rewards to go
        '''
        Gs=np.zeros_like(rewards) #shape:[num steps,1]
        # for t in range(len(rewards)):
        #     Gsum=0
        #     for i in range(t,len(rewards)):
        #         Gsum += self.discount**i * rewards[i]
        #     Gs[t] = Gsum
        #faster:
        Gsum = 0
        for i in reversed(range(0,len(rewards))):
            Gsum = Gsum * discount + rewards[i]
            Gs[i] = Gsum
        mean = np.mean(Gs)
        std = np.std(Gs)
        advantage = (Gs-mean)# Advantage = dicounted Returns - baseline
        advantage /= (std+1e-7)#normalize; 1e#7 weil std 0 sein kann

        return advantage



    def oneHot(self,action):
        action_onehot = np.zeros(self.action_space)
        action_onehot[action] = 1
        return action_onehot


    def plot(self,df,save=False):
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.set_ylabel('Steps',)
        ax.yaxis.label.set_color('y')
        ax2 = ax.twinx()
        ax2.set_ylabel('Reward')
        ax2.yaxis.label.set_color('c')
        # ax.grid(which='minor',axis='y')
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
        handles = handles1 + handles2
        labels=[l.get_label() for l in handles]
        plt.legend(handles, labels,loc=3)
        ax.set_xlabel('Episodes')
        ax.grid(visible=True, which='major',axis='both', color='#666666', linestyle='-', alpha=0.5)
        # Show the minor grid lines with very faint and almost transparent grey lines
        ax.minorticks_on()
        ax.grid(visible=True, which='minor',axis='both', color='#999999', linestyle='-', alpha=0.2)
        plt.tight_layout()
        plt.show()


    def logMetrics(self,episode,score,steps):
        self.df.loc[episode,'score']=score
        self.df.loc[episode,'steps']=steps
        self.df.loc[episode,'score_avg'] =self.df['score'].rolling(100,min_periods=1).mean().iloc[-1]
        self.df['score_mean'] =self.df['score'].mean()
        self.df.loc[episode,'steps_avg'] =self.df['steps'].rolling(100,min_periods=1).mean().iloc[-1]

        # if tune.is_session_enabled():
        #     tune.report(steps=steps,
        #                 score=score,
        #                 score_avg=self.df.loc[episode,'score_avg'],
        #                 episode=episode)	#log to ray tune

        # if wandb.run:
        #     wandb.log({'steps':steps,
        #                 'score':score,
        #                 'score_avg':self.df.loc[episode,'score_mean'],
        #                 'episode':episode})
        #     #if best avg performance save model
        #     if self.df.loc[episode,'score_avg'] >= self.df['score_avg'].max():
        #         #
        #         # #self.Actor.save('.//Models//bestModel.h5')
        #         # #wandb.save('bestModel.h5')
        #         print('Saving Model to',wandb.run.dir)
        #         self.Actor.save_weights(os.path.join(wandb.run.dir, "bestModel_weights.h5"))
        #         wandb.save(os.path.join(wandb.run.dir, "bestModel_weights.h5"))

        if episode % 100 == 0:
            #self.plot(self.df)
            pass

    def run(self,checkpoints = False, start=1): # train only when episode is finished
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_space[0]])
        done, score, = False, 0
        states, actions, action_probs, rewards, next_states, dones = [], [], [], [], [], []
        self.episode = 0
        for i in range(start,self.config['episodes']) :
            #print('\nEpisode %s/%s'%(i,config['episodes']))
            state, done, score = self.env.reset(), False, 0
            state = np.reshape(state, [1, self.state_space[0]])
            steps =0
            while not done:

                # Actor picks an action
                action,  probs = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_space[0]])
                # Memorize (state, action, reward) for training
                #self.updateMemory(state, action, probs, reward, next_state, done)
                states.append(state)
                actions.append(self.oneHot(action))
                action_probs.append(probs)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                # Update current state
                state = next_state
                score += reward
                steps += 1
                # if not wandb.run:
                #     if i%5 ==0:
                #         #self.env.render()
                #         pass

                if done:
                    #create checkpoint for pbt
                    # if checkpoints:
                    #     if i%10 ==0:
                    #         with tune.checkpoint_dir(step=i) as checkpoint_dir:
                    #             path = os.path.join(checkpoint_dir, "checkpoint")
                    #             with open(path, "w") as f:
                    #                 f.write(json.dumps({"episode": i}))
                    #             #print('Saving checkpoint to: ', checkpoint_dir)
                    #             path = os.path.join(checkpoint_dir, "model.h5")
                    #             #print('Saving model to: ', path)
                    #             self.Actor.save_weights(path)

                    self.logMetrics(i,score,steps)

                    if len(states) >= self.trainingBatchSize:
                        print('\nEpisode %s/%s\n'%(i,self.config['episodes']),  '- steps: %s, score %s, average: %s '%(steps,score,self.df['score_avg'].iloc[-1]))
                        self.train(states, actions, action_probs, rewards, next_states, dones)
                        states, actions, action_probs, rewards, next_states, dones = [], [], [], [], [], []
            self.episode += 1
        #wandb.finish()



    def testAgent(self):
        path =os.path.join(os.getcwd(),'Models','PG','LL','bestModel_weights.h5')
        try:
            self.Actor.load_weights(path)
        except:
            print('Model not found at: ',path)
            return
        print('loaded Model from: ',path)
        scores = []
        for i in range(10):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                state = np.reshape(state, [1, self.state_space[0]])
                action = self.act(state,test=True)
                new_state, reward, done, _ = self.env.step(action)
                score += reward
                state = new_state
                self.env.render()
            print("episode: {}/{}, score: {:.3f}".format(i+1, 10, score))
            scores.append(score)
        print('Average score: ',np.mean(scores))
        self.env.close()



if __name__ == "__main__":

    config={'env':'MountainCar-v0','episodes': 10_000,'discount':0.999,'lr':0.0001,
          'layer1':64,'layer2':64,'layer3':None,'activation':'elu','batchsize':1000,
          'entropy':0.003,'alpha':0.6}
    #wandb.init(config=config, entity="meisterich", project="AKI", group ='LunarLanderWind',mode='online',job_type='PG')
    # env = gym.make('LunarLander-v2',enable_wind=True)
    # agent = PGAgent(config,env=env)
    #start normal training:
    #agent.run()

    #test Agent with best model:
    env = gym.make('LunarLander-v2',enable_wind=True)
    agent = PGAgent(config,env=env)
    agent.testAgent()
  















    # #from here for hyperparameter tuning:
    # @wandb_mixin
    # def startTraining(config, checkpoint_dir=None):

    #     wandb.init(config=config, entity="meisterich", project="AKI", group ='LunarLanderPG',mode='online',job_type='pbt')
    #     env = gym.make('LunarLander-v2',enable_wind=True)
    #     agent = PGAgent(config,env)
    #     start =1
    #     if checkpoint_dir:
    #         path = os.path.join(checkpoint_dir, "model.h5")
    #         agent.Actor.load_weights(path)
    #         with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
    #             state = json.loads(f.read())
    #             start = state["episode"] + 1

    #     agent.run(checkpoints = True, start=start)
    #     print('out of run()')
    #     #agent.plot(agent.df)
    #     print('Done')
    #     return agent.df

    # def startHyperparameterSearch(config):
    #     config["wandb"]= {'project': 'AKI','api_key':'fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1','group':'LunarLanderPG','job_type':'pbt','mode':'online'}
    #     #Hyperparameter search:
    #     parameters = {'lr':tune.loguniform(0.0001, 0.1),'activation':tune.choice(['tanh','elu']),
    #                 'entropy':tune.qloguniform(0.0005, 0.01,0.0001),'alpha':tune.quniform(0.1, 0.9, 0.1),
    #                 'batchsize':tune.quniform(500, 3000,100),}

    #     #use for asha scheduler
    #     # config['lr'] = parameters['lr']
    #     # config['trajectories_in_Batch'] = parameters['trajectories_in_Batch']
    #     # config['activation'] = parameters['activation']
    #     # config['entropy'] = parameters['entropy']
    #     # config['alpha'] = parameters['alpha']
    #     # #doesnt work with pbt
    #     # config['layer1'] = parameters['layer1']
    #     # config['layer2'] = parameters['layer1']
    #     # config['layer3'] = parameters['layer3']

    #     #asha: Async Successive Halving Algorithm aka async Hyperband
    #     asha =ASHAScheduler(metric="score_avg",mode="max",time_attr='episodes',grace_period=4000,max_t=config['episodes'])

    #     # pbt = PopulationBasedTraining(time_attr='episodes',metric="score_avg",mode="max",perturbation_interval=50,burn_in_period=500,
    #     #                             hyperparam_mutations={'lr':parameters['lr'],'trajectories_in_Batch':parameters['trajectories_in_Batch'],
    #     #                             'entropy':parameters['entropy'],'alpha':parameters['alpha']})

    #     # pbt = PB2(time_attr='episode',metric="score_avg",mode="max",perturbation_interval=50,
    #     #             hyperparam_bounds={"lr": [0.0001, 0.001],'entropy':[0.0001, 0.1],'alpha':[0.1, 0.9],'trajectories_in_Batch':[1,5]},
    #     #         )



    #     ray.init(address=None)
    #     #starts hyperparameter search with
    #     analysis = tune.run(startTraining,
    #                         config=config,
    #                         num_samples=30,
    #                         scheduler=asha,
    #                         resources_per_trial={'cpu': 1},
    #                         progress_reporter=CLIReporter(max_report_frequency=30),
    #                         fail_fast=True,
    #                         checkpoint_score_attr="score_avg",
    #                         local_dir="ray_results/",
    #                         sync_config=tune.SyncConfig(),
    #                         #checkpoint_freq=10,
    #                         keep_checkpoints_num=1,
    #                         resume="AUTO",
    #                         # callbacks=[WandbLoggerCallback(project = "AKI",group ='LunarLanderPG',api_key="fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1",
    #                         #                                  mode='online',job_type='pbt',log_config=True)],
    #                         )
    #     # ray.shutdown()
    #     print("Best hyperparameters found were: ", analysis.get_best_config("score_avg",'max'))
    #     dfs = analysis.fetch_trial_dataframes()
    #     ax = None  # This plots everything on the same plot
    #     for d in dfs.values():
    #         ax = d.plot('episodes','score_avg',ax=ax)
    #     plt.ylabel('reward_avg')
    #     plt.xlabel('Episodes')
    #     plt.show()

    # startHyperparameterSearch(config)


