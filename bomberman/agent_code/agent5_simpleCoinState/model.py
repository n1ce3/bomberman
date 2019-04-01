"""
This class defines the model
"""

import tensorflow as tf
import numpy      as np

import random

class Model:
    def __init__(self, stateSize, numActions, batchSize):
        self.stateSize  = stateSize
        self.numActions = numActions
        self.batchSize  = batchSize
        # initialize states
        self.states     = None
        # initialize actions
        self.actions    = None
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
        layer1         = tf.layers.dense(self.states, 300, activation=tf.nn.relu)
        #layer2         = tf.layers.dense(layer1, 256, activation=tf.nn.relu)
        # output layer
        self.outLayer  = tf.layers.dense(layer1, self.numActions)
        # loss
        loss           = tf.losses.mean_squared_error(self.qSA, self.outLayer)
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
        # batch training step
        session.run(self.optimizer, feed_dict={self.states: xBatch, self.qSA: yBatch})

class Memory:
    def __init__(self, maxMemory):
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
