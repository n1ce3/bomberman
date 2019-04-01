"""
This class defines the model
"""

import numpy as np
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import random

class TreeQModel:
    def __init__(self, actions, stateSize, numTestGames, isTraining):
        # test settings
        self.numTestGames = numTestGames
        self.isTraining   = isTraining


        # Constants
        self.maxEpsilon   = 1.0
        self.minEpsilon   = 0.001
        self.epsilon      = self.maxEpsilon
        self.gamma        = 0.95
        self.alpha        = 0.5
        self.actions      = actions
        self.stateSize    = stateSize

        # setup the model
        self.model        = RandomForestRegressor(n_estimators=1, min_samples_split=10, min_samples_leaf=10, random_state=0)
        # setup PCA
        self.pca = PCA(n_components = 20)
        # Initialize model
        self.initModel()

        # setup memory
        self.maxMemory    = 5000000
        self.memory       = Memory(self.maxMemory)

    def initModel(self):
        X       = np.random.rand(self.pca.get_params()['n_components']+1, self.stateSize)
        targets = np.zeros((self.pca.get_params()['n_components']+1, len(self.actions)))
        self.pca.fit(X)
        X = self.pca.transform(X)
        self.model.fit(X, targets)

    def chooseAction(self, state, isTraining, gameState):
        if isTraining & (random.random() < self.epsilon):
            # return random action
            if random.random() < 0.2:
                return self.findNearestCoin(gameState['self'], gameState['coins'], gameState['arena'])
            else:
                return random.choice(self.actions)
        else:
            # predict for current state
            try:
                state = self.pca.transform(state.reshape(1, -1))
            except:
                print('Error in transform while choosing action.')
                state = self.pca.transform(state.reshape(1, -1))
            pred   = self.model.predict(state.reshape(1, -1))[0]
            action = self.actions[np.argmax(pred)]
            return action

    def updateEpsilon(self, episode, nRounds):
        trainEpisodes = nRounds-(nRounds/(self.numTestGames+100))*self.numTestGames
        self.epsilon = self.maxEpsilon - episode*(self.maxEpsilon - self.minEpsilon)/(trainEpisodes-1)

    def replay(self):
        batch   = self.memory.sample()
        # prepare for fit
        X       = np.zeros((len(batch), self.pca.get_params()['n_components']))
        X_large = np.zeros((len(batch), self.stateSize))
        targets = np.zeros((len(batch), len(self.actions)))

        start = time.time()
        # update Q Values
        i = 0
        for state, action, reward, nextState in batch:
            X_large[i] = state

            state = self.pca.transform(state.reshape(1, -1))

            qSA     = self.model.predict(state.reshape(1, -1))[0]
            if nextState is None:
                qUpdate = self.alpha*reward
            else:
                # predict Q-Values for current state
                # calculate new values

                nextState = self.pca.transform(nextState.reshape(1, -1))

                qSAD    = self.model.predict(nextState.reshape(1, -1))[0]
                # Q-Update
                qUpdate = qSA[action] + self.alpha*(reward + self.gamma * np.amax(qSAD) - qSA[action])

            qSA[action] = qUpdate
            X[i]        = state
            targets[i]  = qSA
            i += 1

        end = time.time()
        print('Time to iterate over whole batch: {0:.3f} s'.format(end-start))
        start = time.time()
        # fit for dimension trasnform
        self.pca.fit(X_large)

        self.model.fit(X, targets)
        self.isFit = True
        end = time.time()
        print('Time used to fit: {0:.3f} s'.format(end-start))

    def findNearestCoin(self, pos, coins, arena):

        # calculate distances
        dist = []
        #print('now')
        #print('Coins: {}'.format(coins))
        #print(gameState['coins'])
        #print(gameState['Coins'])
        for coin in coins:
            c = np.array([coin[0], coin[1]])
            a = np.array([pos[0], pos[1]])
            dist.append(np.linalg.norm(c-a))

        nearestCoin = coins[np.argmin(dist)]

        # calculate angle to agent clockwise
        ang = np.arctan2(nearestCoin[0]-pos[0], nearestCoin[1]-pos[1])
        angle = np.rad2deg(-(ang - np.pi/2.) % (2 * np.pi))

        col, row = pos[0], pos[1]
        # return best direction to go
        if (0 <= angle < 45) | (315 <= angle < 360):
            if (arena[row][col+1]==-1) & (angle < 45):
                return 'DOWN'
            elif (arena[row][col+1]==-1) & (angle >= 315):
                return 'UP'
            else:
                return 'RIGHT'
        elif 45 <= angle < 135:
            if (arena[row+1][col]==-1) & (angle < 90):
                return 'RIGHT'
            elif (arena[row+1][col]==-1) & (angle >= 90):
                return 'LEFT'
            else:
                return 'DOWN'
        elif 135 <= angle < 225:
            if (arena[row][col-1]==-1) & (angle < 180):
                return 'DOWN'
            elif (arena[row][col-1]==-1) & (angle >= 180):
                return 'UP'
            else:
                return 'LEFT'
        elif 225 <= angle < 315:
            if (arena[row-1][col]==-1) & (angle < 270):
                return 'LEFT'
            elif (arena[row-1][col]==-1) & (angle >= 270):
                return 'RIGHT'
            else:
                return 'UP'


class Memory:
    def __init__(self, maxMemory):
        self.maxMemory = maxMemory
        self.samples   = []

    def addSample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.maxMemory:
            self.samples.pop(0)

    def sample(self):
        # shuffle data
        random.shuffle(self.samples)
        return self.samples
