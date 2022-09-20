import os

from pandas import value_counts
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1:cpu, 0:first gpu
import random
import gym
import pylab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() 
import copy

# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler,PopulationBasedTraining
# from ray.tune import Trainable
# from ray.tune.integration.wandb import (WandbLoggerCallback,WandbTrainableMixin,wandb_mixin,)
# import wandb
# from wandb.keras import WandbCallback
# WANDB_API_KEY='fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1'



class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, config,env=None):
        # Initialization
        # Environment and PPO parameters
        print('run eagerly:',tf.executing_eagerly())
        if env is None:
            print('making env',config['env'])
            self.env_name = config['env']
            self.env = gym.make(self.env_name)#,continuous=True)#,enable_wind=True)
        else:
            print('using env')
            self.env = env
        self.config = config
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape
        self.EPISODES = config['episodes'] # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = config['lr']
        self.epochs = 20 # training epochs
        self.shuffle=False
        self.episodeBatchSize = config['episodeBatchSize'] # number of steps per episode before training
        self.batch_full = False
        self.state_memory = []
        self.action_memory = []
        self.action_probs_memory = []
        self.reward_memory = []
        self.done_memory = []
        self.next_state_memory = []
        self.df =pd.DataFrame()

        
        layerSizes = [config['layer1'],config['layer2'],config['layer3']]
        self.optimizer = Adam

        self.replay_count = 0
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        #self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        #self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Actor, self.Critic = self.build_model(layerSizes, lr=self.lr, optimizer = self.optimizer)
        


    def build_model(self, layerSizes, lr, optimizer, activation='tanh', shared_model=False):
        X_input = Input(self.state_space)
        

        if shared_model ==False:
            X = Dense(layerSizes[0], activation="tanh", kernel_initializer=tf.random_normal_initializer(stddev=0.5))(X_input)
            X = Dense(layerSizes[1], activation="tanh", kernel_initializer=tf.random_normal_initializer(stddev=0.5))(X)
            policy = Dense(self.action_space, activation="softmax")(X)
    
            self.Actor = Model(inputs = X_input, outputs = policy)
            self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))
            self.Actor.summary()

            X = Dense(layerSizes[0], activation="tanh", kernel_initializer=tf.random_normal_initializer(stddev=0.5))(X_input)
            X = Dense(layerSizes[1], activation="tanh", kernel_initializer=tf.random_normal_initializer(stddev=0.5))(X)
            value = Dense(1, activation="linear")(X)

            self.Critic = Model(inputs = X_input, outputs = value)
            self.Critic.compile(loss='mse', optimizer=optimizer(lr=lr))
            self.Critic.summary()
        
            return self.Actor, self.Critic

        if shared_model == True:
            X = Dense(64, activation="elu", kernel_initializer=tf.random_normal_initializer(stddev=0.05))(X_input)
            X = Dense(64, activation="elu", kernel_initializer=tf.random_normal_initializer(stddev=0.05))(X)
            #X = Dense(64, activation="elu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
            output = Dense(self.action_space, activation="softmax")(X)
            value = Dense(1, activation="linear")(X)

            self.Actor = Model(inputs = X_input, outputs = output)
            self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))
            self.Actor.summary()

            self.Critic = Model(inputs = X_input, outputs = value)
            self.Critic.compile(loss='mse', optimizer=optimizer(lr=lr))
            self.Critic.summary()

            return self.Actor, self.Critic

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, pdfs, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = self.config['entropy']
        
        prob = actions * y_pred#y_pred is the probability of the action with current policy
        old_prob = actions * pdfs

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def act(self, state,test=False):
        probs = self.Actor.predict(state)[0]
        if test:
            action = np.argmax(probs)
            return action
        action = np.random.choice(self.action_space, p=probs)
        return action, probs


    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [reward + gamma * (1 - done) * new_value - value for reward, done, new_value, value in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        #advantages = np.zeros_like(deltas)
        advantages = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas)-1)):
            advantages[t] = advantages[t] + (1 - dones[t]) * gamma * lamda * advantages[t + 1]

        target = advantages + values
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return np.vstack(advantages), np.vstack(target)

    def train(self, states, actions, action_probs, rewards, next_states, dones):
        print("Training...")
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        #print(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        action_probs = np.vstack(action_probs)
        
        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages
        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
    
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, action_probs, actions])
        #y_true_critic = np.hstack([target, values])
        # training Actor and Critic networks
        a_loss = self.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True)
        c_loss = self.Critic.fit(states,target, epochs=self.epochs, verbose=0, shuffle=True)

        #print(self.state_memory)
        self.replay_count += 1
 

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

    def logMetrics(self,episode,score,steps,checkpoints=False):
        self.df.loc[episode,'score']=score
        self.df.loc[episode,'steps']=steps
        self.df.loc[episode,'score_avg'] =self.df['score'].rolling(100,min_periods=1).mean().iloc[-1]
        self.df['score_mean'] =self.df['score'].mean()

        # if tune.is_session_enabled():
        #         tune.report(steps=steps,
        #                     score=score,
        #                     score_avg=self.df.loc[episode,'score_avg'],
        #                     episode=episode)	#log to ray tune	

        # if wandb.run:
        #     wandb.log({'steps':steps,
        #                 'score':score,
        #                 'score_avg':self.df.loc[episode,'score_avg'],
        #                 'episode':episode})

        #     #if best avg performance save model
        #     if self.df.loc[episode,'score_avg'] >= self.df.loc[episode,'score_avg'].max():
        #         print('Saving Model')
        #         self.Actor.save_weights(os.path.join(wandb.run.dir, "bestModel_weights.h5"))
        #         wandb.save(os.path.join(wandb.run.dir, "bestModel_weights.h5"))
    
    def run(self): # train only when episode is finished
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_space[0]])
        done, score = False, 0
        episode =0
        while episode < self.EPISODES:
            states, actions, action_probs, rewards, next_states, dones = [], [], [], [], [], []
            print('collecting expirience...')
            state, done, score = self.env.reset(), False, 0
            state = np.reshape(state, [1, self.state_space[0]])
            steps =0
            for t in range (self.episodeBatchSize):
                #self.env.render()
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
                steps +=1
                if done:
                    episode +=1
                    self.logMetrics(episode,score,steps)
                    print("episode: {}/{}, score: {}, average: {:.2f} ".format(episode, self.EPISODES, score, self.df.loc[episode,'score_avg']))
                    state, done, score,steps = self.env.reset(), False, 0,0
                    state = np.reshape(state, [1, self.state_space[0]])

            self.train(states, actions, action_probs, rewards, next_states, dones)
                        
        self.env.close()

        

    def testAgent(self):
        path =os.path.join(os.getcwd(),'Models','PPO','LL','bestModel_weights.h5')
        try:
            self.Actor.load_weights(path)
        except:
            print('Model weights not found at: ',path)
            return
        print('loaded Model weights from: ',path)
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

    config={'env':'LunarLander-v2','episodes': 10000,'discount':0.999,'lr':0.0001,
         'layer1':64,'layer2':64,'layer3':None,'entropy':0.001,'episodeBatchSize':1000,}
    #wandb.init(config=config, entity="meisterich", project="AKI", group ='LunarLanderPPO',mode='online')

    # env = gym.make('LunarLander-v2',enable_wind=True)
    # agent = PPOAgent(config,env)
    #agent.run() 

    #test Agent with best model:
    env = gym.make('LunarLander-v2',enable_wind=True)
    agent = PPOAgent(config,env)
    agent.testAgent()