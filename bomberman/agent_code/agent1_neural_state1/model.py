"""
This class defines the model which gets trained. It is also used
to determine the actions if the game is played without training.
"""
# import modules
import tensorflow as tf
import numpy      as np
import random

# import informations from the game
from settings import e, events, settings

# define model class
class Model:
    def __init__(self, actions, stateSize, trainEpisodes, isTraining):
        # trainings flag
        self.isTraining = isTraining

        self.stateSize  = stateSize
        self.numRuns    = trainEpisodes
        self.actions    = actions
        self.numActions = len(actions)

        # training constants
        self.maxEpsilon = 1
        self.minEpsilon = 0.01
        self.epsilon    = self.maxEpsilon
        # strength of exponential decay
        self.decay      = 0.000025
        # discount
        self.gamma      = 0.95
        # learning rate
        self.alpha      = 0.6

        # set memory
        self.batchSize  = 10
        self.memory     = Memory(50000)
        # size of layer
        self.numUnits   = 550
        # initialize states
        self.states     = None
        # initialize output layer
        self.outLayer   = None
        self.optimizer  = None
        # global variables
        self.varInit    = None
        # setup the model
        self.defineModel()

    def defineModel(self):
        # set placeholder for states
        self.states    = tf.placeholder(shape=[None, self.stateSize], dtype=tf.float32)
        # set placeholder for Q(s,a)
        self.qSA       = tf.placeholder(shape=[None, self.numActions], dtype=tf.float32)
        # create neural network
        # hidden layer
        layer1         = tf.layers.dense(self.states, self.numUnits, activation=tf.nn.relu)
        #layer2         = tf.layers.dense(layer1, 256, activation=tf.nn.relu)
        # output layer
        self.outLayer  = tf.layers.dense(layer1, self.numActions)
        # loss function
        loss           = tf.losses.mean_squared_error(self.qSA, self.outLayer)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)
        # initialize global variables
        self.varInit   = tf.global_variables_initializer()

    def predictOne(self, state, session):
        # predict for one state
        return session.run(self.outLayer, feed_dict={self.states: state.reshape(1, self.stateSize)})

    def predictBatch(self, states, session):
        # predict for whole batch of states
        return session.run(self.outLayer, feed_dict={self.states: states})

    def trainBatch(self, session, xBatch, yBatch):
        # batch training
        session.run(self.optimizer, feed_dict={self.states: xBatch, self.qSA: yBatch})

    def chooseAction(self, state, session, gameState):
        # if random number is smaller than epsilon a random action will be returned,
        # due to exploration. A small fraction of random moves are replaced by a hint
        # to find the next coin since coin selections are rare events in the beginning.
        if self.isTraining & (random.random() < self.epsilon):
            # return random action
            if random.random() < 0.2:
                action = self.findNearestCoin(gameState['self'], gameState['coins'], gameState['arena'])
                if action is not None:
                    return action
                else:
                    return random.choice(self.actions)
            else:
                return random.choice(self.actions)
        # if not random then just use the net to predict next step
        else:
            return self.actions[np.argmax(self.predictOne(state, session))]

    def findNearestCoin(self, pos, coins, arena):
        # methode to find the optimal direction to the nearest coin.
        # as long as coins are vivible at the moment
        if len(coins) != 0:
            # calculate distances
            dist = []
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
        # if no coins are visible at the moment. Go back and do a random action
        else:
            return None

    def updateEpsilon(self, episode):
        # updates the epsilon value after each episode
        # the epsilon value decays exponentially over the episodes.
        self.epsilon = self.expEpsilon(episode, self.numRuns)

    def expEpsilon(self, x, n):
        # evaluate exponential function at current episode to get next epsilon
        b = (self.minEpsilon-self.maxEpsilon*np.exp(-self.decay*n))/(1-np.exp(-self.decay*n))
        a = self.maxEpsilon - b
        return a*np.exp(-self.decay*x)+b

    def replay(self, session):
        # replay the saved episodes until buffer is empty
        # the states from the buffer are used to do q-learning
        # therefore the q values are getting updated due to the
        # update rule (see more in report)
        numBatches = 0
        while True:
            # choose bstchSize random samples from buffer
            batch = self.memory.sample(self.batchSize)
            # break training if memory is empty
            if batch==None:
                break
            # while memory not empty use all samples in memory for training
            else:
                # choose states from batch
                states     = np.array([val[0] for val in batch])
                nextStates = np.array([(np.zeros(self.stateSize) if val[3] is None else val[3]) for val in batch])

                # predict Q(s,a) given the batch of states
                qSA        = self.predictBatch(states, session)
                # predict Q(s',a')
                qSAD       = self.predictBatch(nextStates, session)

                # setup training arrays
                x = np.zeros((len(batch), self.stateSize))
                y = np.zeros((len(batch), self.numActions))

                for i, b in enumerate(batch):
                    state, action, reward, nextState = b[0], b[1], b[2], b[3]
                    # get the current q values for all actions in state
                    currentQ = qSA[i]
                    # update the q value for action
                    if nextState is None:
                        # in this case, the game completed after action, so there is no max Q(s',a')
                        # prediction possible
                        currentQ[action] = self.alpha*reward
                    else:
                        # update q-value
                        currentQ[action] = currentQ[action] + self.alpha*(reward + self.gamma * np.amax(qSAD[i]) - currentQ[action])
                    x[i] = state
                    y[i] = currentQ
                # now train batch with updated Q values
                self.trainBatch(session, x, y)
                numBatches +=1
        # print out how many batches have been trained.
        print(numBatches, ' batches trained')

# handle memory
class Memory:
    # this class handles just the samples in the memory
    def __init__(self, maxMemory):
        # buffer
        self.maxMemory = maxMemory
        self.samples   = []

    def addSample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.maxMemory:
            self.samples.pop(0)

    def sample(self, numSamples):
        # shuffle data
        random.shuffle(self.samples)
        # return batch
        if numSamples > len(self.samples):
            # after this batch memory will be empty
            if len(self.samples)==0:
                return None
            else:
                batch = self.samples[:len(self.samples)]
                del self.samples[:numSamples]
                return batch
        else:
            # return numSamples first samples
            batch = self.samples[:numSamples]
            # remove batch from memory
            del self.samples[:numSamples]
            return batch
