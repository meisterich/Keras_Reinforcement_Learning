'''
PPO Algorithm for continuous action space
with action standard deviation from model output
'''


from ast import Lambda
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
from tensorflow.keras.layers import Input, Dense,concatenate, Lambda
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
    def __init__(self, input_shape, action_space, lr, optimizer,clip_value=0.2,layers=[64,64],entropy=0.001):
        self.clip_value = clip_value
        self.entropy = entropy
        X_input = Input(input_shape)
        self.action_space = action_space
        
        X = Dense(layers[0], activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        #X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(layers[1], activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        if len(layers)>2:
            X = Dense(layers[2], activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        mu = Dense(self.action_space, activation="tanh")(X)
        log_sigma = Dense(self.action_space, activation="softplus")(X)
        log_sigma =Lambda(lambda x: x *-1.0)(log_sigma)
        mu_lsigma = concatenate([mu, log_sigma])

        self.Actor = Model(inputs = X_input, outputs = mu_lsigma)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))
        print(self.Actor.summary())

    def ppo_loss_continuous(self, y_true, y_pred):
        '''
        Loss function for PPO algorithm

        y_true: [action, advantage]
        y_pred: [mu, sigma]
        '''
        #print('loss')
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = self.clip_value
        mu = y_pred[:, :self.action_space]
        log_sigma = y_pred[:, self.action_space:]
        
        # print('y_pred:',y_pred)
        # print('mu:',mu)
        # print('sigma:',sigma)
        logp = self.gaussian_likelihood(actions, mu, log_sigma)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        loss_entropy = self.entropy * K.mean(-(K.log(2*np.pi*K.square(K.exp(log_sigma)))+1) / 2) 

        return actor_loss + loss_entropy

    def gaussian_likelihood(self, actions, pred, log_std): # for keras custom loss
        #print('likelihood')
        log_std = log_std * np.ones(self.action_space,dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        mu_lsigma = self.Actor.predict(state,verbose=0)
        mu = mu_lsigma[:, :self.action_space]
        log_sigma = mu_lsigma[:, self.action_space:]
        log_sigma = log_sigma *np.ones(self.action_space,dtype=np.float32)
        
        return mu, log_sigma


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer,clip_value=0.2):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))
        self.clip_value = clip_value

        V = Dense(128, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        #V = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        V = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = self.clip_value
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
    

class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self,config,env):
        # Initialization
        # Environment and PPO parameters
        print('running eagerly:',tf.executing_eagerly())
        self.config = config
        self.env = env
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.EPISODES = config['Episodes']# total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.lr = config['lr']
        self.lr_critic = config['lr_critic']
        self.epochs = config['epochs'] # training epochs
        self.shuffle = True
        self.Training_batch = int(config['batchsize'])
        #self.mini_batch = int(config['miniBatchsize'])
        self.gamma = config['discount'] # discount factor
        self.lamda = config['lambda'] # lambda for GAE
        #self.optimizer = RMSprop
        self.optimizer = Adam
        self.df =pd.DataFrame()

        self.replay_count = 0

        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size,
                                 lr=self.lr, optimizer = self.optimizer,clip_value =config['clip_value'],
                                 layers=config['layers'],entropy=config['entropy'])
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr_critic,
                                 optimizer = self.optimizer)
        
        self.old_actor_weights = self.Actor.Actor.get_weights()
        self.tau = config['tau']
        if config['load']:
            path =os.path.join(os.getcwd(),'Models','PPO_continuous','BW_h','bestModel_weights.h5')
            self.Actor.Actor.load_weights(path)
            print("Actor Model Loaded from: ",path)



    # def get_std(self, log_sigma):
    #     log_std = -log_sigma*np.ones(self.action_size,dtype=np.float32)
    #     return np.exp(log_std)


    def act(self, state,test=False):
        # Use the network to predict the next action to take, using the model
        mu,log_std = self.Actor.predict(state)
        std = np.exp(log_std)
        #print('mu:',mu)
        #print('sigma:',sigma)
        if test==True:
            action = mu
            return action

        low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
        action = mu + np.random.uniform(low, high, size=mu.shape) * std

        #action = pred + np.random.normal(0, self.std, size=pred.shape)
        #action = np.clip(action, low, high)
        
        logp_t = self.gaussian_likelihood(action, mu, log_std)

        return action, logp_t,mu,std


    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def discount_rewards(self, reward):#gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True):
        '''
        Generalized Advantage Estimation
        
        Args:
            rewards: list of rewards
            dones: list of done flags
            values: list of state values
            next_values: list of next state values
            gamma: discount factor
            lamda: GAE factor
            normalize: normalize the result

        '''
        gamma =self.gamma
        lamda = self.lamda
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def train(self, states, actions, rewards, dones, next_states, logp_ts):
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
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)
        values=np.vstack(values)
        # Compute discounted rewards and advantages
        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        target = np.vstack(target)
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        y_true = np.hstack([advantages, actions, logp_ts])
        
        # training Actor and Critic networks
        self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        self.soft_update_weights()
        
        self.replay_count += 1
 
    def soft_update_weights(self):
        """Softupdate of the target network.
        In ppo, the updates of the 
        """
        
        weights = np.array(self.Actor.Actor.get_weights())
        old_weights = np.array(self.old_actor_weights)
        new_weights = self.tau*weights + (1-self.tau)*old_weights
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

        time.sleep(0.1)
        # score_avg = self.df.loc[episode,'score_avg100']
        # if wandb.run:
        #     wandb.log({'steps':steps,
        #                 'score':score,
        #                 'score_avg':self.df.loc[episode,'score_avg'],
        #                 'score_avg100':self.df.loc[episode,'score_avg100'],
        #                 'episode':episode})
        #     #if best avg performance save model
        #     if self.df.loc[episode,'score_avg100'] >= self.df['score_avg100'].max():
        #         print('Saving Model')
        #         self.Actor.Actor.save_weights(os.path.join(wandb.run.dir, "bestModel_weights.h5"))
        #         wandb.save(os.path.join(wandb.run.dir, "bestModel_weights.h5"))

        # if episode %500 ==0:# and (not tune.run):
        #     #self.plot(self.df)
        #     pass

        # if tune.is_session_enabled():
        #         tune.report(steps=steps,
        #                     score=score,
        #                     score_avg=score_avg,
        #                     episode=episode)	#log to ray tune
        # #log checkpoints for PBT
        # if checkpoints:
        #     if episode%10 ==0:
        #         with tune.checkpoint_dir(step=episode) as checkpoint_dir:
        #             path = os.path.join(checkpoint_dir, "checkpoint")
        #             with open(path, "w") as f:
        #                 f.write(json.dumps({"episode": episode}))
        #             #print('Saving checkpoint to: ', checkpoint_dir)
        #             path = os.path.join(checkpoint_dir, "ActorModel.h5")
        #             #print('Saving model to: ', path)
        #             self.Actor.Actor.save_weights(path)
        #             path = os.path.join(checkpoint_dir, "CriticModel.h5")
        #             self.Critic.Critic.save_weights(path)
    
    def run(self, checkpoints = False,start=0):
        self.episode=start
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, dones, logp_ts = [], [], [], [], [], []
            steps=0
            mus,sigmas=[],[]
            print('collecting experience')
            for t in range(self.Training_batch):
                if self.episode % 2 == 0:
                    self.env.render()
                    pass
                # Actor picks an action
                action, logp_t,mu,sigma = self.act(state)
                mus.append(mu)
                sigmas.append(sigma)
                # Retrieve new state, reward, and whether the state is terminal
                #print('action',action)
                next_state, reward, done, _ = self.env.step(action[0])
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                # Update current state shape
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                steps+=1
                if done:
                    self.episode += 1
                    self.logMetrics(self.episode, score, steps, checkpoints)
                    print("episode: {}/{},steps: {} score: {:.3f}, average: {:.2f}".format(self.episode, self.EPISODES,steps, score, self.df['score_avg100'].iloc[-1]))
                    
                    state, done, score,steps = self.env.reset(), False, 0,0
                    state = np.reshape(state, [1, self.state_size[0]])

            # if self.episode % 5 == 0:
            #     self.printActionHistogramm(actions,mus,sigmas)
            self.train(states, actions, rewards, dones, next_states, logp_ts)
            if self.episode >= self.EPISODES:
                break

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
        path =os.path.join(os.getcwd(),'Models','PPO_continuous',env,'bestModel_weights.h5')
        try:
            self.Actor.Actor.load_weights(path)
        except:
            print('Model not found at: ', path)
            return
        print('loaded Model from: ', path)
        print('Testing Agent')
        time.sleep(1)
        scores = []
        for i in range(10):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                state = np.reshape(state, [1, self.state_size[0]])
                action= self.act(state,test=True)
                #print('action',action)
                new_state, reward, done, _ = self.env.step(action[0])
                score += reward
                state = new_state
                self.env.render()
            print("episode: {}/{}, score: {:.3f}".format(i+1, 10, score))
            scores.append(score)
        print('Average score: ',np.mean(scores))
        self.env.close()


if __name__ == "__main__":
    config={'Episodes': 20_000,'discount':0.99,'lr':0.0001,'lr_critic':0.0003,
            'batchsize':1024,'clip_value':0.2,'lambda':0.9,'epochs':20,
            'layers':[64,64],'tau':0.9,'entropy':0.002,'load':True}
    # wandb.init(config=config, entity="meisterich", project="AKI", group ='BipedalWalker_hardcore',mode='online',job_type='PPO_v3_pretrained')
    # env = gym.make('BipedalWalker-v3',hardcore=True)
    # agent = PPOAgent(config,env)
    # agent.run() # train as PPO
    #wandb sync --sync-all --include-synced


#test Environments with best model
def test():
    #wait for keyboard input
    inp= input('which environment to test?\n type:\n LL: LunarLander\n BW: BipedalWalker\n BW_h: BipedalWalker_Hardcore')
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
    agent = PPOAgent(config,env)
    agent.testAgent(env=inp)

test()















    
    # #from here for hyperparameter tuning:
    # def startTraining(config=None, checkpoint_dir=None):
    #     wandb.login(key='fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1')
    #     wandb.init(config=config, entity="meisterich", project="AKI", group ='BipedalWalker_hardcore',mode='online',job_type='PPO_v3_pretrained')
    #     config = wandb.config
    #     print('config',config)
    #     env = gym.make('BipedalWalker-v3',hardcore=True)
    #     agent = PPOAgent(config,env)
    #     start =1
    #     ckpts = False
    #     agent.run(checkpoints = ckpts, start=start)
    #     print('out of run()')
    #     #agent.plot(agent.df)
    #     print('Done')
    #     return agent.df

    # def startHyperparameterSearch(config):
    #     #config["wandb"]= {'project': 'AKI','api_key':'fa46257ffe8e2bd1ecfc8a16f754bff2ecc727e1','group':'LunarLanderPPO_continuous','job_type':'pbt','mode':'online'}
    #     #Hyperparameter search:
    #     parameters = {'lr':tune.loguniform(0.0001, 0.001),'lr_critic':tune.loguniform(0.0001, 0.005),
    #                     'batchsize':tune.quniform(512,3072,128),
    #                     'clip_value':tune.choice([0.1,0.2,0.3]),'lambda':tune.loguniform(0.8, 0.99),}

    #     #use for asha scheduler
    #     config['lr'] = parameters['lr']
    #     config['lr_critic'] = parameters['lr_critic']
    #     config['batchsize'] = parameters['batchsize']
    #     config['clip_value'] = parameters['clip_value']
    #     config['lambda'] = parameters['lambda']



    #     #asha: Async Successive Halving Algorithm aka async Hyperband
    #     asha =ASHAScheduler(metric="score_avg",mode="max",time_attr='episode',grace_period=3000,max_t=config['Episodes'])

    #     # pbt = PopulationBasedTraining(time_attr='episodes',metric="score_avg",mode="max",perturbation_interval=50,burn_in_period=500,
    #     #                             hyperparam_mutations={'lr':parameters['lr'],'trajectories_in_Batch':parameters['trajectories_in_Batch'],
    #     #                             'entropy':parameters['entropy'],'alpha':parameters['alpha']})

    #     pbt = PB2(time_attr='episode',metric="score",mode="max",perturbation_interval=100,
    #                 hyperparam_bounds={"lr": [0.0001, 0.001],'lr_critic':[0.001,0.01],
    #                 #'log_std':[-2.0,-0.1],
    #                 'clip_value':[0.1, 0.3],
    #                 })



    #     ray.init(address=None)
    #     #starts hyperparameter search with
    #     analysis = tune.run(startTraining,
    #                         config=config,
    #                         num_samples=30,
    #                         scheduler=asha,
    #                         resources_per_trial={'gpu': 1},
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


    # def startHyperparameterSearchWANDB(config):
        
    #     sweep_config = {
    #         "name" : 'PPOcontiBW', "method" : "bayes",
    #         'metric': {'goal': 'maximize', 'name': 'score_avg100'},
    #         'early_terminate':{'type':'hyperband','max_iter': 10_000,'s':1},
    #         'parameters':{
    #             'batchsize':{'distribution': 'q_uniform', 'min': 512, 'max': 2048, 'q': 256},
    #             'lr':{'distribution': 'q_log_uniform', 'min': math.log(0.00005),'max': math.log(0.0002),'q': 0.00001},
    #             'lr_critic':{'distribution': 'log_uniform', 'min': math.log(0.0001),'max': math.log(0.0005)},
    #             'clip_value':{'distribution': 'q_uniform', 'min': 0.1, 'max': 0.3, 'q': 0.05},
    #             'lambda':{'distribution': 'q_uniform', 'min': 0.8,'max': 0.99,'q': 0.01},
    #             'tau':{'distribution': 'q_uniform', 'min': 0.5, 'max': 0.9, 'q': 0.1},
    #             }}
    #     sweep_config['parameters'].update({
    #         'Episodes': {'value': config['Episodes']},
    #         'discount': {'value': config['discount']},
    #         'epochs': {'value': config['epochs']},
    #         'load': {'value': config['load']},
    #     })
    #     sweep_id = wandb.sweep(sweep_config, project='AKI')
    #     wandb.agent(sweep_id, startTraining,count=30)
    #     return



    # startHyperparameterSearchWANDB(config)