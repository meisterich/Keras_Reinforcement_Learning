import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1:cpu, 0:first gpu; cpu faster on laptop?
import random
import gym
print('gym version:', gym.__version__)
import pylab
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() # usually using this for fastest performance
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense,concatenate
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras import backend as K
import copy
import time

# import ray
# from ray import tune
# #print('rayversion:', ray.__version__)
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler,PopulationBasedTraining
# from ray.tune.schedulers.pb2 import PB2
# from ray.tune.logger import DEFAULT_LOGGERS
# import wandb
# from wandb.keras import WandbCallback
# WANDB_API_KEY='fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1'




class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space
        
        X = Dense(64, activation="relu")(X_input)#kernel_initializer=tf.random_normal_initializer(stddev=0.05)
        #X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu")(X)
        mu = Dense(self.action_space, activation="tanh")(X)
        sigma = Dense(self.action_space, activation="softplus")(X)
        mu_sigma = concatenate([mu, sigma])

        self.Actor = Model(inputs = X_input, outputs = mu_sigma)
        self.Actor.compile(loss=self.pg_loss_continuous, optimizer=optimizer(lr=lr))
        #print(self.Actor.summary())

    def pg_loss_continuous(self, y_true, y_pred):
        '''
        Loss function for PPO algorithm

        y_true: [action, advantage]
        y_pred: [mu, sigma]
        '''
        print('loss')
        returns, actions,dones = y_true[:, :1], y_true[:, 1:1+self.action_space],y_true[:, 1+self.action_space:-1]
 
        mu = y_pred[:, :self.action_space]
        sigma = y_pred[:, self.action_space:]
        print('y_pred:',y_pred)
        print('mu:',mu)
        print('sigma:',sigma)
        logp = self.gaussian_likelihood(actions, mu, K.log(sigma))

        actor_loss = -K.sum(logp * returns)
        # done_nr = K.sum(dones)
        # if done_nr < 1.0:
        #     done_nr = 1.0
        # actor_loss /=done_nr

        loss_entropy = 0.0005 * K.mean(-(K.log(2*np.pi*K.square(sigma))+1) / 2) 

        return actor_loss + loss_entropy

    def gaussian_likelihood(self, actions, pred, log_std): # for keras custom loss
        print('likelihood')
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.Actor.predict(state,verbose=0)



class PGAgent:
    # PPO Main Optimization Algorithm
    def __init__(self,config,env):
        # Initialization
        # Environment and PPO parameters
        print('running eagerly:',tf.executing_eagerly())
        self.config = config
        
        self.env = env
        
        
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.shape[0]
        print('action_size:',self.action_size)
        print('state_size:',self.state_size)
        self.EPISODES = int(config['Episodes'])# total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.lr = config['lr']
        self.Training_batch = int(config['batchsize'])
        self.mini_batch = int(config['miniBatchsize'])
        self.gamma = config['discount'] # discount factor
        self.shuffle = True

        #self.optimizer = RMSprop
        self.optimizer = Adam
        self.df =pd.DataFrame()

        self.replay_count = 0

        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
 
        self.old_actor_weights = self.Actor.Actor.get_weights()
        self.alpha = config['alpha']




    def act(self, state,test=False):
        # Use the network to predict the next action to take, using the model
        #print('act')
        state =np.array(state)
        #print('state shape',state.shape)
        #print(state)
        state = np.expand_dims(state,axis=0)
        #print('state shape',state.shape)
        #print(state)
        mu_sigma = self.Actor.predict(state)
        mu = mu_sigma[:, :self.action_size]
        sigma = mu_sigma[:, self.action_size:]
        if test:
            action = mu
            return action
        low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
        action = np.random.normal(loc=mu, scale=sigma)
        #action = np.clip(action, low, high)
        return action, mu,sigma


    def discount_rewards(self, reward,dones):#gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma*(1-dones[i]) + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r


    def train(self, states, actions, rewards, dones, next_states):
        '''
        Train the Actor-Critic network
        Args:
            states: list of states
            actions: list of actions
            rewards: list of rewards
            dones: list of done flags
            next_states: list of next states
            logp_ts: list of log probabilities of actions
            '''

        print('training')
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        dones = np.vstack(dones)
        returns = self.discount_rewards(rewards,dones)
        returns = np.vstack(returns)
        y_true = np.hstack([returns, actions, dones])
        
        # training Actor and Critic networks
        self.Actor.Actor.fit(states, y_true, batch_size=self.mini_batch, epochs=10, verbose=0, shuffle=self.shuffle)
        
        self.soft_update_weights()
        
        self.replay_count += 1
 
    def soft_update_weights(self):
        """Softupdate of the target network.
        In ppo, the updates of the 
        """
        
        weights = np.array(self.Actor.Actor.get_weights())
        old_weights = np.array(self.old_actor_weights)
        new_weights = self.alpha*weights + (1-self.alpha)*old_weights
        self.Actor.Actor.set_weights(new_weights)
        self.old_actor_weights = new_weights


    def plot(self,df,save=False):
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.set_ylabel('Steps',)
        ax.yaxis.label.set_color('y')
        ax2 = ax.twinx()
        ax2.set_ylabel('Reward')
        ax2.yaxis.label.set_color('c')
        # ax.grid(which='minor',axis='y')
        df.plot(y=['score','score_avg100','score_avg'],ax=ax2,color=['c','blue','tab:blue'],legend=False)
        df.plot(y=['steps'],color='y',ax=ax,legend=False)

        #fig.suptitle(' lr: %3s, discount: %5s'%(lr,discount))
        # title = str(self.__class__.__name__)+' '+str(self.config['env'])
        # title += ('\n lr=%s, discount=%s,' %(self.config['lr'],self.config['discount']))
        # layerSizes=self.layerSizes
        # title += (' Net Layout %s'%layerSizes)
        fig.suptitle('plot')
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

    def logMetrics(self,episode,score,steps,checkpoints=False):
        #print(episode,score,steps)
        self.df.loc[episode,'score']=score
        self.df.loc[episode,'steps']=steps
        self.df.loc[episode,'score_avg100'] =self.df['score'].rolling(100,min_periods=1).mean().iloc[-1]
        self.df.loc[episode,'score_avg'] =self.df['score'].mean()
        self.df.loc[episode,'steps_avg'] =self.df['steps'].rolling(100,min_periods=1).mean().iloc[-1]

        # time.sleep(0.1)
        # score_avg = self.df.loc[episode,'score_avg']
        # if wandb.run:
        #     wandb.log({'steps':steps,
        #                 'score':score,
        #                 'score_avg':score_avg,
        #                 'episode':episode})
        #     #if best avg performance save model
        #     if self.df.loc[episode,'score_avg'] >= self.df['score_avg'].max():
        #         print('Saving Model')
        #         self.Actor.Actor.save_weights(os.path.join(wandb.run.dir, "bestModelweights.h5"))
        #         wandb.save(os.path.join(wandb.run.dir, "bestModelweights.h5"))

        # if episode %50 ==0:# and (not tune.run):
        #     #self.plot(self.df)
        #     pass

        # if tune.is_session_enabled():
        #         tune.report(steps=steps,
        #                     score=score,
        #                     score_avg=score_avg,
        #                     episode=episode)	#log to ray tune
    

    def run(self, checkpoints = False,start=0):
        states, next_states, actions, rewards, dones = [], [], [], [], []
        for episode in range(self.EPISODES):
            done, score,steps=False, 0,0
            try:
                state,_ = self.env.reset()
            except:
                state = self.env.reset()
            print('collecting experience')

            while not done:
                if self.episode % 2 == 0:
                    #self.env.render()
                    pass
                # Actor picks an action
                action,mu,sigma = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                #print('action',action)
                try:
                    next_state, reward, done, _,_ = self.env.step(action[0])
                except:
                    next_state, reward, done, _ = self.env.step(action[0])
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                # Update current state shape
                state = next_state
                score += reward
                steps+=1
                
            self.logMetrics(episode, score, steps, checkpoints)
            print("episode: {}/{},steps: {} score: {:.3f}, average: {:.2f}".format(episode, self.EPISODES,steps, score, self.df['score_avg100'].iloc[-1]))
            #train only when collecter enough samples
            if len(states) >= self.Training_batch:
                self.train(states, actions, rewards, dones, next_states)
                states, next_states, actions, rewards, dones = [], [], [], [], []

        self.env.close()
        #wandb.finish()

    def printActionHistogramm(self,actions,mus,sigmas):
        actions =np.array(actions).squeeze()
        mus =np.array(mus).squeeze()
        sigmas =np.array(sigmas).squeeze()
        print('actions shape: ',actions.shape)
        #print(actions)
        #print(actions[:,0])
        bins = np.linspace(-1, 1, 100)
        fig, ax = plt.subplots()
        ax.hist(mus[:,0], bins=bins)
        plt.title('mu[0](main) Histogramm')
        plt.show()
        fig, ax = plt.subplots()
        ax.hist(mus[:,1], bins=bins)
        plt.title('mu[1](lateral) Histogramm')
        plt.show()

        ig, ax = plt.subplots()
        ax.hist(sigmas[:,0], bins=bins)
        plt.title('sigma[0](main) Histogramm')
        plt.show()
        fig, ax = plt.subplots()
        ax.hist(sigmas[:,1], bins=bins)
        plt.title('sigma[1](lateral) Histogramm')
        plt.show()

        fig, ax = plt.subplots()
        ax.hist(actions[:,0], bins=bins)
        plt.title('Action[0] Histogramm')
        plt.show()
        fig, ax = plt.subplots()
        ax.hist(actions[:,1], bins=bins)
        plt.title('Action[1] Histogramm')
        plt.show()
        
        time.sleep(3)


    def testAgent(self,env='LL'):
        path =os.path.join(os.getcwd(),'Models','PG_continuous',env,'bestModel_weights.h5')
        try:
            self.Actor.Actor.load_weights(path)
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
                action = self.act(state,test=True)

                new_state, reward, done, _ = self.env.step(action[0])
                score += reward
                state = new_state
                self.env.render()
            print("episode: {}/{}, score: {:.3f}".format(i+1, 10, score))
            scores.append(score)
        print('Average score: ',np.mean(scores))
        self.env.close()



if __name__ == "__main__":
    config={'Episodes': 10_000,'discount':0.99,'lr':0.00005,'batchsize':1000,'alpha':0.9,'miniBatchsize':1000}
    # wandb.init(config=config, entity="meisterich", project="AKI", group ='BipedalWalkerPG',mode='online',job_type='sigma')
    # env = gym.make('BipedalWalker-v3')
    # agent = PGAgent(config,env)
    # agent.run() 

#test Environments with best model
def test():
    #wait for keyboard input
    inp= input('which environment to test?\n type:\n LL: LunarLander\n BW: BipedalWalker\n')
    if inp == 'LL':
        env = gym.make('LunarLanderContinuous-v2',enable_wind=True)
        print('start testing LunarLander') 
    elif inp == 'BW':
        env = gym.make('BipedalWalker-v3',hardcore=False)
        print('start testing BipedalWalker')
    time.sleep(1)
    agent = PGAgent(config,env)
    agent.testAgent(env=inp)

test()