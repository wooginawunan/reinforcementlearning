from __future__ import print_function

import pickle
import glob
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from model import Model
from utils import *
import random
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm

def read_data(datasets_dir="./data", frac = 0.1, training_samples=100000, includemine =True):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    files = glob.glob(os.path.join(datasets_dir, 'data*.pkl.gzip'))
    data = {}
    for data_file in files:
        if includemine or ((not includemine) and ('2020' not in data_file)):
            f = gzip.open(data_file,'rb')
            data_ = pickle.load(f)
            for k, v in data_.items():
                if k in data:
                    data[k].extend(v)
                else:
                    data[k] = v

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)][:training_samples], y[:int((1-frac) * n_samples)][:training_samples]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1, classification=False):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    def transform(X_data):
        frames, _, _, _ = X_data.shape
        
        X_data_ = []
        for i in tqdm.tqdm(range(history_length-1, frames)):
            X = []
            for j in range(i-history_length+1, i+1):
                X.append(rgb2gray(X_data[j, :]))
            X_data_.append(np.stack(X, 0))

        return np.stack(X_data_, 0)

    print("... preprocessing")
    X_train = transform(X_train)
    y_train = y_train[history_length-1:]
    X_valid = transform(X_valid)
    y_valid = y_valid[history_length-1:]
    if classification:
        y_valid = np.array([action_to_id(x) for x in y_valid])
        y_train = np.array([action_to_id(x) for x in y_train])
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, 
                history_length, 
                n_minibatches, 
                batch_size, 
                lr, 
                model_dir="./models",
                classification=True,
                resnet=False,
                moddrop=False,
                shared=False,
                gpu=False,
                ):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")

    # TODO: specify your neural network in model.py 
    agent = Model(history_length, classification, resnet, moddrop, shared)

    if gpu: agent.to(torch.device('cuda:0'))
    if classification:
        weight = torch.tensor([1, 5, 5, 5, 5] )
        #weight = torch.tensor([1, 1, 1, 1, 1] )
        weight = weight.type(torch.FloatTensor)
        criterion = nn.CrossEntropyLoss(weight=weight.cuda())
    else:
        criterion = nn.MSELoss()

    optimizer = optim.SGD(agent.parameters(), lr=lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.1)

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    running_loss = 0
    max_val_acc = 0
    for i in tqdm.tqdm(range(n_minibatches)):

        agent.train()
        training_indices = list(range(len(y_train)))
        random.Random(i).shuffle(training_indices)

        X = X_train[training_indices[:batch_size]]
        y = y_train[training_indices[:batch_size]]

        X = torch.from_numpy(X)
        if gpu: 
            X = X.to(torch.device('cuda:0'))
        y_hat = agent(X)

        optimizer.zero_grad()
        y = torch.from_numpy(y).to(torch.device('cuda:0'))

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        ll = float(loss)
        running_loss += ll

        if i % 9 == 0:
            
            neptune.log_metric('training loss', ll)
            neptune.log_metric('running loss', running_loss/(i+1))

            if classification:
                _, y_hat = y_hat.max(1)
                neptune.log_metric('training acc', (y_hat == y).float().mean())

            agent.eval()    
            val_indices = list(range(len(y_valid)))
            random.Random(i).shuffle(val_indices)

            X = X_valid[val_indices[:batch_size]]
            y = y_valid[val_indices[:batch_size]]
            y = torch.from_numpy(y).to(torch.device('cuda:0'))

            X = torch.from_numpy(X)
            if gpu: 
                X = X.to(torch.device('cuda:0'))

            y_hat = agent(X)

            loss = criterion(y_hat, y)
            neptune.log_metric('val loss', float(loss))
            neptune.log_metric('step', i)
            if classification:
                _, y_hat = y_hat.max(1)
                val_acc = (y_hat == y).float().mean()
                neptune.log_metric('val acc', val_acc)

                if max_val_acc<val_acc:
                    max_val_acc = val_acc
                    torch.save(agent.state_dict(), os.path.join(model_dir, "agent.pt"))

    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":

    import neptune

    # The init() function called this way assumes that
    # NEPTUNE_API_TOKEN environment variable is defined.

    parser = argparse.ArgumentParser()

    parser.add_argument("--historylength", type=int, default=1)
    parser.add_argument("--n-minibatches", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--trainingsize", type=int, default=1000000)
    parser.add_argument("--classification", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--resnet", action="store_true", default=False)
    parser.add_argument("--moddrop", action="store_true", default=False)
    parser.add_argument("--shared", action="store_true", default=False)
    parser.add_argument("--notincludemine", action="store_false", default=True)

    args = parser.parse_args()

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data", 0.1, args.trainingsize, args.notincludemine)

    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    neptune.init('wooginawunan/RLhws')
    neptune.create_experiment(name='%s'%name,
        params={'historylength': args.historylength, 
        'batch_size': args.batch_size,
        'n_minibatches': args.n_minibatches,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'trainingsize': len(y_train),
        'classification': args.classification,
        'resnet': args.resnet,
        'moddrop': args.moddrop,
        'shared': args.shared
        })

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, 
        history_length=args.historylength, classification=args.classification)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, 
        history_length=args.historylength,
        n_minibatches=args.n_minibatches, 
        batch_size=args.batch_size, 
        lr=args.lr,
        model_dir="./models/%s"%name,
        classification=args.classification,
        resnet=args.resnet,
        moddrop=args.moddrop,
        shared=args.shared,
        gpu=args.gpu)
 
