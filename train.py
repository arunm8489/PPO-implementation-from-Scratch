import os 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import gym
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from agent import *
from memory import *
from actor_criric_models import *
import warnings
warnings.simplefilter("ignore")




def plot_learning_curve(x, scores, figure_file):
    """ 
    function for plotting training curve
    """
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == "__main__":

        ### directory for saving models ###
        if not os.path.exists('tmp'):
            os.makedirs('tmp')      


        env = gym.make('CartPole-v0')
        N = 20
        batch_size = 5
        n_epochs = 4
        alpha = 0.0003
        agent = Agent(state_dim=env.observation_space.shape,
                      action_dim=env.action_space.n, 
                      batch_size=batch_size,
                      n_epochs=n_epochs,
                      policy_clip=0.2,
                      gamma=0.99,lamda=0.95, 
                      adam_lr=alpha)
        n_games = 3
        figure_file = 'cartpole.png'
        best_score = env.reward_range[0]
        score_history = []
        learn_iters = 0
        avg_score = 0
        n_steps = 0
        for i in range(n_games):
            current_state,info = env.reset()
            terminated,truncated = False,False
            done = False
            score = 0
            while not done:
                action, prob, val = agent.choose_action(current_state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = 1 if (terminated or truncated) else 0
                n_steps += 1
                score += reward
                agent.store_data(current_state, action, prob, val, reward, done)
                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1
                current_state = next_state
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                    'time_steps', n_steps, 'learning_steps', learn_iters)


        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)

