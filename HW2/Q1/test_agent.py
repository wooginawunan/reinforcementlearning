from __future__ import print_function

import argparse
from datetime import datetime
import numpy as np
import gym
import os
import json
import neptune
from model import Model
import torch
from utils import *


def run_episode(env, agent, historylength, rendering=True, 
    classification=False,
    max_timesteps=1000,
    gpu=False,
    ):
    
    agent.eval()
    episode_reward = 0
    step = 0

    history = []
    state = env.reset()
    images = []

    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        img = rgb2gray(state)
        history.append(img)
        if len(history)>historylength:
            history = history[1:]

        #neptune.log_image('state', img)
        images.append(img)
        
        X = []

        j = 0
        for i in range(historylength):
            if i < historylength-len(history):
                X.append(np.zeros([96, 96]))
            else:
                X.append(history[j])
                j +=1
        X = np.stack([np.stack(X, 0)], 0)
         
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        # a = ...
        X = torch.from_numpy(X)

        X = X.type(torch.FloatTensor)

        if gpu: 
            X = X.to(torch.device('cuda:0'))

        if gpu:     
            a = agent(X).detach().cpu().numpy()[0]

        else:
            a = agent(X).detach().numpy()[0]
            print(a)   
            a = a.argmax()
            print(a) 

         
        if classification:
            a = id_to_action(a)
            print(a)
        #print(a)
        if step<1:
            next_state, r, done, info = env.step([0, 1, 0])   
        else:
            next_state, r, done, info = env.step(a) 
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward, images


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)                 
    rendering = True
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelname")
    parser.add_argument("--historylength", type=int, default=1)
    parser.add_argument("--classification", action="store_true", default=False,)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--resnet", action="store_true", default=False)
    parser.add_argument("--moddrop", action="store_true", default=False)
    parser.add_argument("--shared", action="store_true", default=False)

    args = parser.parse_args()

    n_test_episodes = 15                  # number of episodes to test

    # neptune.init('wooginawunan/RLhws')
    # neptune.create_experiment(name='testing_%s'%args.modelname,
    #     params={'model': args.modelname, 
    #     })


    # TODO: load agent
    agent = Model(args.historylength, args.classification, args.resnet, args.moddrop, args.shared)

    checkpoint = torch.load(os.path.join('./models', args.modelname, "agent.pt"), 
        map_location=torch.device('cpu'))
    agent.load_state_dict(checkpoint)

    env = gym.make('CarRacing-v0').unwrapped
    results = dict()
    episode_rewards = []
    for i in range(n_test_episodes):

        episode_reward, images = run_episode(env, agent, 
            historylength=args.historylength, 
            rendering=rendering,
            classification=args.classification,
            gpu=args.gpu)

        images = np.array(images)
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        np.save("results/results_bc_agent-%s.npy"% time_now, images)
        episode_rewards.append(episode_reward)
        print(i, 'reward:', episode_reward)
        # save results in a dictionary and write them into a .json file
        
        results["episode_rewards"] = episode_rewards
        results["mean"] = np.array(episode_rewards).mean()
        results["std"] = np.array(episode_rewards).std()
     
        fname = "results/results_bc_agent-%s.json" % time_now
        fh = open(fname, "w")
        json.dump(results, fh)
            
    env.close()
    print('... finished')
