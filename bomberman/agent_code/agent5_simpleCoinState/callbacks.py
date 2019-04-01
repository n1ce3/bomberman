
import numpy as np
import tensorflow as tf
import random

from settings import e, events, settings
from .model   import Model, Memory

def setup(agent):
    """
    Here we set up our Agent. This function is called before the first step of the game.
    """
    # possible actions
    agent.actions     = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    agent.action      = None

    # state
    agent.stateSize   = int(settings['cols']**2-2*settings['cols']-2*(settings['cols']-2)-((settings['cols']-3)/2)**2)
    agent.state       = None
    agent.preState    = None

    agent.episode     = 1

    # setup model
    setupModel(agent)

def setupModel(agent):
    # is trainings mode
    agent.isTraining  = True
    # set model
    agent.batchSize   = 10
    agent.numActions  = len(agent.actions)
    agent.model       = Model(agent.stateSize, agent.numActions, agent.batchSize)
    # saver
    agent.saver       = tf.train.Saver()
    # start tensorflow session
    agent.session     = tf.Session()
    # initialize global variables
    agent.session.run(agent.model.varInit)
    # restore model
    #agent.saver.restore(agent.session, 'model')
    # training constants
    agent.epsilon     = 0.3
    agent.gamma       = 0.95
    agent.alpha       = 0.1
    agent.maxT        = 200
    agent.minT        = 1
    agent.temperature = agent.maxT
    agent.testEpisode = 0
    agent.testAfter   = 1000
    agent.numTestGames= 100
    agent.testReward  = 0
    agent.globalEpisode = 1
    agent.preReward   = 0

    # rewards
    agent.totalReward = 0
    agent.reward      = {'MOVED_LEFT':          -1,
                         'MOVED_RIGHT':         -1,
                         'MOVED_UP':            -1,
                         'MOVED_DOWN':          -1,
                         'WAITED':              -5,
                         'INTERRUPTED':         -1,
                         'INVALID_ACTION':     -10,
                         'BOMB_DROPPED':        -1,
                         'BOMB_EXPLODED':       -1,
                         'CRATE_DESTROYED':     30,
                         'COIN_FOUND':           0,
                         'COIN_COLLECTED':     300,
                         'KILLED_OPPONENT':     60,
                         'KILLED_SELF':       -300,
                         'GOT_KILLED':        -300,
                         'OPPONENT_ELIMINATED':100,
                         'SURVIVED_ROUND':       0}
    # save training
    agent.memory       = Memory(50000)
    agent.fileName     = 'eGreedy300MB.txt'
    agent.testFileName = 'eGreedyTest300MB.txt'
    if agent.isTraining:
        loadFile(agent)
        loadTestFile(agent)

    agent.logger.info(f'Training model is set')

def saveToFile(agent):
    agent.file.write('{}\t{}\n'.format(agent.episode, agent.totalReward/100.))
    agent.file.flush()

def loadFile(agent):
    agent.file = open(agent.fileName, 'w')

def loadTestFile(agent):
    agent.testFile = open(agent.testFileName, 'w')

def saveTestToFile(agent):
    agent.testFile.write('{}\t{}\n'.format(agent.episode-1, agent.testReward/agent.numTestGames))
    agent.testFile.flush()

def act(agent):
    """
    This function is called at each step of the game and determines what our Agent will do next.
    Therefore the state in which the agent is gets loaded. The corresponding action will be
    done regarding to the Q-Table. If game is in training mode the agent will do random actions
    due to explore the enviroment. The fraction of random moves is determined by epsilon.
    """
    agent.state       = getCurrentState(agent)
    agent.action      = chooseAction(agent)
    agent.next_action = agent.action

def reward_update(agent):
    """
    This function is called after each move.
    Update rewards here after each move.
    """
    agent.logger.info('REWARD {}'.format(agent.events))
    # count reward
    reward = 0
    for r in agent.events:
        reward += agent.reward[events[r]]

    agent.preReward += reward

    if (agent.game_state['step'] > 2) & agent.isTraining:
        agent.memory.addSample((agent.preState, agent.actions.index(agent.preAction), agent.preReward, agent.state))
        agent.totalReward += agent.preReward
    else:
        agent.testReward  += agent.preReward

    # safe action, state and gained reward
    agent.preAction = agent.action
    agent.preState  = agent.state
    agent.preReward = reward

def end_of_episode(agent):
    """
    """
    if agent.isTraining:
        # give reward for last:
        reward = 0
        for r in agent.events:
            reward += agent.reward[events[r]]

        agent.preReward += reward

        agent.memory.addSample((agent.preState, agent.actions.index(agent.preAction), agent.preReward, None))
        agent.totalReward += agent.preReward

        if agent.episode%100==0:
            print('T:\t\t ', agent.temperature)
            print('Total Reward:\t ', agent.totalReward/100.)
            replay(agent)
            saveToFile(agent)
            print('Trained Episodes ', agent.episode-99, ' - ', agent.episode)
            # reset total reward for next 10 episodes
            agent.totalReward = 0

        # update temperature

        updateT(agent)

        # stop training to test
        if (agent.episode)%(agent.testAfter)==0:
            # play games pause training
            # turn train mode off
            agent.isTraining = False
            if not (agent.globalEpisode== settings['n_rounds']):
                print('\ntraining stopped')
                print('TEST')
            else:
                print('Test file closed')
                agent.testFile.close()

        # next episode starts now
        agent.episode += 1
        agent.logger.info('End of Episode: {}'.format(agent.episode-1))

    else:
        # count test games
        agent.testEpisode += 1
        agent.testReward  += agent.preReward
        # if test at end continue training and save data to file
        if agent.testEpisode==agent.numTestGames:
            # save net after test
            agent.saver.save(agent.session, 'model2')
            print('model saved!')
            # save test reward
            saveTestToFile(agent)
            # rest test
            agent.testEpisode = 0
            agent.testReward  = 0
            print('training continued\n')
            # continue training
            agent.isTraining = True

    # add up global episode to determine if game is done
    agent.globalEpisode += 1
    agent.collectedCoins = 0

def getCurrentState(agent):
    """
    Get current state of the Game. A state is defined as a vector whose entries are the state of each tile.
    A tile can either be emtpy (0), a wall (-1) or contains a coin (1).

    First state representation:
        just consider tiles that can change their state. So don't consider walls.
    """
    # create numpy array of current state
    # get arena
    arena = agent.game_state['arena']
    # get coins
    coins = agent.game_state['coins']
    # get position
    pos   = agent.game_state['self']

    # create current state
    currentState = []

    # take arena and flatten without outter wall to minimize state size since the walls always remain the same
    for x,y in np.ndindex(arena.shape):
        tile      = arena[x,y]
        tileState = []
        if tile != -1:
            # check if empty (0), coin (1) or pos (-1)
            if (y,x) in coins:
                tileState.append(1)
            elif (y,x) == (pos[0], pos[1]):
                tileState.append(-1)
            else:
                tileState.append(0)
            # append to state
            currentState += tileState
        # don't consider walls
        else:
            continue
    return np.array(currentState)

def chooseAction(agent):
    Q = agent.model.predictOne(agent.state, agent.session)[0]
    if agent.isTraining:
        # with prob epsilon act randomly
        if random.random() < agent.epsilon:
            return random.choice(agent.actions)
        # with prob 1-epsilon choose from max-boltzmann distribution
        else:
            A  = agent.numActions
            T  = agent.temperature
            pi = np.exp(Q/T)/np.sum(np.exp(Q/T))
            return np.random.choice(agent.actions, p=pi)
    else:
        # predict for current state
        action = agent.actions[np.argmax(Q)]
        return action

def replay(agent):
    numBatches = 0
    while True:
        batch      = agent.memory.sample(agent.model.batchSize)
        # break training if memory is empty
        if batch==None:
            break
        # while memory not empty use all samples in memory for training
        else:
            states     = np.array([val[0] for val in batch])
            nextStates = np.array([(np.zeros(agent.model.stateSize) if val[3] is None else val[3]) for val in batch])
            # predict Q(s,a) given the batch of states
            qSA        = agent.model.predictBatch(states, agent.session)
            # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
            qSAD       = agent.model.predictBatch(nextStates, agent.session)
            # setup training arrays
            x = np.zeros((len(batch), agent.model.stateSize))
            y = np.zeros((len(batch), agent.model.numActions))
            for i, b in enumerate(batch):
                state, action, reward, nextState = b[0], b[1], b[2], b[3]
                # get the current q values for all actions in state
                currentQ = qSA[i]
                # update the q value for action
                if nextState is None:
                    # in this case, the game completed after action, so there is no max Q(s',a')
                    # prediction possible
                    currentQ[action] = agent.alpha*reward
                else:
                    currentQ[action] = currentQ[action] + agent.alpha*(reward + agent.gamma * np.amax(qSAD[i]) - currentQ[action])
                x[i] = state
                y[i] = currentQ
            agent.model.trainBatch(agent.session, x, y)
            numBatches +=1
    print(numBatches, ' batches trained')

def updateT(agent):
    trainEpisodes = settings['n_rounds']-(settings['n_rounds']/(agent.testAfter+agent.numTestGames))*agent.numTestGames
    agent.temperature = agent.maxT - agent.episode*(agent.maxT - agent.minT)/(trainEpisodes-1)
