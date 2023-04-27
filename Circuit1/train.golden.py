import os

import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
import parl
import numpy as np

import sys

from parl.utils import logger, summary, ReplayMemory

from parl.algorithms import DDPG,OAC ,TD3

from sim import simulate


##OAC MODEL
# clamp bounds for Std of action_log
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0

N = 128


class OACModel(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(OACModel, self).__init__()
        self.actor_model = OACActor(obs_dim, action_dim)
        self.critic_model = OACCritic(obs_dim, action_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class OACActor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(OACActor, self).__init__()
        #N = 24#256
        self.l1 = nn.Linear(obs_dim, N)
        self.l2 = nn.Linear(N, N)
        self.mean_linear = nn.Linear(N, action_dim)
        self.std_linear = nn.Linear(N, action_dim)

    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))

        act_mean = self.mean_linear(x)
        act_std = self.std_linear(x)
        act_log_std = paddle.clip(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return act_mean, act_log_std


class OACCritic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(OACCritic, self).__init__()

        #N = 24 #256
        # Q1 network
        self.l1 = nn.Linear(obs_dim + action_dim, N)
        self.l2 = nn.Linear(N, N)
        self.l3 = nn.Linear(N, 1)

        # Q2 network
        self.l4 = nn.Linear(obs_dim + action_dim, N)
        self.l5 = nn.Linear(N, N)
        self.l6 = nn.Linear(N, 1)

    def forward(self, obs, action):
        x = paddle.concat([obs, action], 1)

        # Q1
        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2
        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2



####AI Model
class ACModel(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(ACModel, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(obs_dim, action_dim)
    def policy(self, obs):
        return self.actor_model(obs)
    def value(self, obs, action):
        return self.critic_model(obs, action)
    def get_actor_params(self):
        return self.actor_model.parameters()
    def get_critic_params(self):
        return self.critic_model.parameters()

class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        #N = 24
        self.l1 = nn.Linear(obs_dim, N)
        self.l2 = nn.Linear(N, N)
        self.l3 = nn.Linear(N, action_dim)

    def forward(self, obs):
        a = F.relu(self.l1(obs))
        a = F.relu(self.l2(a))
        return paddle.tanh(self.l3(a))

class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        #N = 24
 
        self.l1 = nn.Linear(obs_dim, N)
        self.l2 = nn.Linear(N + action_dim, N)
        self.l3 = nn.Linear(N, 1)

    def forward(self, obs, action):
        q = F.relu(self.l1(obs))
        q = F.relu(self.l2(paddle.concat([q, action], 1)))
        return self.l3(q)



######TD3MOdel

    
class TD3MODEL(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(TD3MODEL, self).__init__()
        self.actor_model = TD3Actor(obs_dim, action_dim)
        self.critic_model = TD3Critic(obs_dim, action_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def Q1(self, obs, action):
        return self.critic_model.Q1(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class TD3Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(TD3Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        action = paddle.tanh(self.l3(x))
        return action


class TD3Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(obs_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(obs_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, obs, action):
        sa = paddle.concat([obs, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, obs, action):
        sa = paddle.concat([obs, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


    


####AI Agent
class 动(parl.Agent):
    def __init__(self, algorithm, act_dim, expl_noise=0.01):
        assert isinstance(act_dim, int)
        super(动, self).__init__(algorithm)

        self.act_dim = act_dim
        self.expl_noise = expl_noise

        self.alg.sync_target(decay=0)

    def sample(self, obs):
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

    def predict(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss



def run_eval_episode(obs, obs_top,obs_bot,agent, rpm , EVAL_EPISODES,EPISODE):
    #env = gym.make('Pendulum-v1', render_mode="human")
    #obs, _ = env.reset()
    total = 0
    #tune K
    #obs = np.array([1.35])
    #obs_top = np.array([2.0])
    #obs_bot = np.array([0.8])

    for e in range(EVAL_EPISODES):
        avg_reward = 0

        for idx in range(EPISODE):

            obs = (obs - (obs_top+obs_bot)/2.0)*2.0/(obs_top - obs_bot)
            action = agent.predict(obs)
            action = action * (obs_top - obs_bot)/2.0 + (obs_bot + obs_top)/2.0
 
            next_state, reward, done = step(action) 
            #print(action, next_state, reward)
            obs = next_state
            avg_reward += reward
        
        avg_reward /= EPISODE
        print("EVAL: ", e, avg_reward)
        total += avg_reward
    return total / EVAL_EPISODES


def step(action):
    #tune K, W6, W11, W8, W12, W2, W1
    #action = np.array([1.35, 1.125, 1.0, 4.0, 4.5, 0.6, 6.0 ]) 
    params = {
        "K": str(action[0]),
#	"K": str(action[0]),
	"W6": str(action[1])+"u",
#        "W6": "1.1277u",
#	"W11":"1000n",
	"W11":str(action[2])+"u",	
	"VN2":"900m",
	"S": "2",
	"VN": "900m",
	"VP": "300m",
#	"W8": "4u",
	"W8": str(action[3])+"u",	
	"F": "1k",
#	"W12":"4.5u",
        "W12": str(action[4])+"u",	
#	"W2":"600n",
        "W2": str(action[5])+"u",		
#	"W1":"6u",
        "W1": str(action[6])+"u",		
	"A":"0.01",
	"L":"245n",
	"W10":"W8",
	"W7":"W6"
    }
    print(params)
    #print("Simulating ")
    DC_gain, Freq, DB3_gain, DB3_Freq = simulate(params)
    print(action, DC_gain, DB3_gain, DB3_Freq, DC_gain*DB3_Freq)
    
    #normalize reward
    GAIN_MIN = 100.0
    reward = (DC_gain - GAIN_MIN) / GAIN_MIN

    next_state = np.array(action)
    return next_state, reward, 0


def run_train_episode(start_obs, obs_top, obs_bot, agent,  rpm , EPISODE,BATCH_SIZE,WARMUP_STEPS):
    
        #obs, _ = env.reset()
        
        episode_reward = 0
        episode_steps = 0
        
        obs = start_obs

        for idx in range(EPISODE):

            obs = (obs - (obs_top+obs_bot)/2.0)*2.0/(obs_top - obs_bot)
 
            if rpm.size() < WARMUP_STEPS:
                #todos (how to sample)
                action = np.random.uniform(-1.0, 1.0, size=action_dim)
            else:
                action = agent.sample(obs)
                #todos (how to sample)
                #todos
            
            action = action * (obs_top - obs_bot)/2.0 + (obs_bot + obs_top)/2.0
            #obs = obs * (obs_top - obs_bot)/2.0 + (obs_bot + obs_top)/2.0

            #new_params = 0.2*action + obs * (obs_top - obs_bot)/2.0 + (obs_bot + obs_top)/2.0
            #new_params = np.clip(new_params, a_min=obs_bot, a_max = obs_top)
            #next_state, reward, done = env.step([action]) 
            #next_state, reward, done = step(new_params) 
            next_state, reward, done = step(action)  
            terminal = 0 # float(done) if episode_steps < env._max_episode_steps else 0 
            
            next_state = next_state.reshape((-1))
            #print("OBS:",obs,action,reward,next_state,terminal, rpm.size())
            rpm.append(obs, action, reward, next_state, terminal)
            obs = next_state
            #reward = (reward + 8.1) / 8.1
            episode_reward += reward

            if rpm.size() >= WARMUP_STEPS:
                for i in range(5):
                    batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                	BATCH_SIZE)
                    agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

            episode_steps += 1

        return episode_reward, episode_steps , obs


if __name__ == "__main__":
    # 定义环境、实例化模型
    #env = gym.make('Pendulum-v1')#, render_mode="human")

    EPOCH = 2000
    EPISODE = 30

    WARMUP_STEPS = 32
    EVAL_EPISODES = 2
    MEMORY_SIZE = int(10000)
    BATCH_SIZE = 12
    GAMMA = 0.99
    TAU = 0.005
    ACTOR_LR = 1e-3
    CRITIC_LR = 1e-3
    EXPL_NOISE = 0.05  # Std of Gaussian exploration noise
    
    PRETRAIN = False
    PRELOAD = False

    #env.reset()
    #action = 1.0#np.random.random()*2 - 1
    #next_state, reward, done,_,_ = env.step([action])    
    #print(action , next_state, reward)

    #obs_dim = env.observation_space.shape[0]
    #action_dim = env.action_space.shape[0]


    ##############################################
    #tune K, W6, W11, W8, W12, W2, W1
    obs = np.array([1.35, 1.127, 1.0, 4.0, 4.5, 0.6, 6.0 ])
    #obs_top = np.array([2.0, 2.0, 2.0, 2.0, 6.0, 2.0 , 10.0 ])
    #obs_bot = np.array([0.5, 0.5, 0.5, 0.5, 3.0, 0.3, 3.0  ])
    obs_top = np.array([1.4, 1.2, 1.1, 4.1, 4.6, 0.7 , 6.10 ])
    obs_bot = np.array([1.3, 1.1, 0.9, 3.9, 4.4, 0.6, 5.90  ])



    obs_dim = len(obs)
    action_dim = len(obs)
    ##############################################


    #DDPG
    #model = ACModel(obs_dim, action_dim)
    #algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)


    #OAC
    model = OACModel(obs_dim, action_dim)
    algorithm = OAC(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, delta = 23.53, alpha = 0.2, beta = 4.66)

    #TD3
    #model = TD3MODEL(obs_dim, action_dim)
    #algorithm = TD3(model, gamma=GAMMA,tau=TAU,actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,  policy_freq=2)


    agent = 动(algorithm, action_dim, expl_noise=EXPL_NOISE)

    
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)


    start_action = 0
    if(PRELOAD):
        if(os.path.exists("rpm.npz")):
            print("load rpm")
            rpm.load("rpm.npz")
            #todos get best start_action from pareto ? 
            #1. 排序reward
            #2. 选最高的reward和对应的action
        
        if(os.path.exists("agent")):
            print("restore agent")
            agent.restore("agent")

    #initial training 
    
    for i in range(10000):
        if(PRETRAIN == False):
            break
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                	BATCH_SIZE)
        agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)
        if(i % 100 == 1):
            print("Pre-training ", i)


    #episode_reward, episode_steps = run_train_episode(agent, env, rpm , EPISODE,BATCH_SIZE,WARMUP_STEPS)

    #run_eval_episode(agent, rpm , EVAL_EPISODES,EPISODE)


    for e in range(EPOCH):
        #train
        #use obs or last_obs?
        episode_reward, episode_steps, last_obs = run_train_episode(obs, obs_top, obs_bot, agent, rpm , EPISODE,BATCH_SIZE,WARMUP_STEPS)
        print("TRAIN: ",e, episode_reward)
        
        #predict 
        if(e % EVAL_EPISODES == 0):
            print("Saving RL agent and data")
            rpm.save("rpm")
            agent.save("agent")

            #predict
            #avg_reward = run_eval_episode(obs, obs_top, obs_bot, agent, rpm , 1,EPISODE)
            #print("EVAL: ",avg_reward)
        
