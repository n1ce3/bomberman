import numpy as np

from settings import e, events, settings
from .model   import *

'''
Train a agent using Q-learning with a regression tree
'''

def setup(agent):
    """
    Here we set up our Agent. This function is called before the first step of the game.
    """
    # possible actions
    agent.actions     = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    agent.action      = None
    agent.preAction   = None

    # state
    agent.stateSize   = int(settings['cols']**2-2*settings['cols']-2*(settings['cols']-2)-((settings['cols']-3)/2)**2)
    agent.state       = None
    agent.preState    = None

    # episode
    agent.episode     = 1

    # setup model
    setupModel(agent)

def setupModel(agent):
    # is trainings mode
    agent.isTraining    = True

    # set model
    agent.batchSize     = 10
    agent.numActions    = len(agent.actions)
    agent.testGames     = 10
    agent.tree          = TreeQModel(agent.actions, agent.stateSize, agent.testGames, agent.isTraining)

    # training episodes
    agent.globalEpisode = 1
    agent.testEpisode   = 0
    agent.coins         = 0

    # rewards
    agent.totalReward   = 0
    agent.testReward    = 0
    agent.preReward     = 0
    agent.rewards       = {'MOVED_LEFT':          -1,
                           'MOVED_RIGHT':         -1,
                           'MOVED_UP':            -1,
                           'MOVED_DOWN':          -1,
                           'WAITED':              -5,
                           'INTERRUPTED':         -1,
                           'INVALID_ACTION':     -10,
                           'BOMB_DROPPED':        -1,
                           'BOMB_EXPLODED':       -1,
                           'CRATE_DESTROYED':     50,
                           'COIN_FOUND':           0,
                           'COIN_COLLECTED':      50,
                           'KILLED_OPPONENT':     60,
                           'KILLED_SELF':       -300,
                           'GOT_KILLED':        -300,
                           'OPPONENT_ELIMINATED':100,
                           'SURVIVED_ROUND':       0}

    # load files
    if agent.isTraining:
        # save training progress
        agent.fileName     = 'eGreedyTree.txt'
        agent.testFileName = 'eGreedyTreeTest.txt'
        loadFile(agent)
        loadTestFile(agent)

def act(agent):
    """
    This function is called at each step of the game and determines what our Agent will do next.
    Therefore the state in which the agent is gets loaded. The corresponding action will be
    done regarding to the Q-Table. If game is in training mode the agent will do random actions
    due to explore the enviroment. The fraction of random moves is determined by epsilon.
    """
    # choose action
    agent.state       = getCurrentState(agent)
    agent.action      = agent.tree.chooseAction(agent.state, agent.isTraining, agent.game_state)
    agent.logger.debug(agent.game_state['coins'])
    agent.next_action = agent.action
    # if training
    if agent.isTraining:
        # don't do this the first step since there is no previous state and no reward
        if agent.game_state['step'] > 1:
            agent.tree.memory.addSample((agent.preState, agent.actions.index(agent.preAction), agent.reward, agent.state))

        # save for next step
        agent.preState  = agent.state
        agent.preAction = agent.action

def reward_update(agent):
    """
    This function is called after each move after consequences are know.
    Is exexcuted at the beginning of each step before action is chosen.
    Update rewards here after each move.
    """
    # count reward
    reward = 0.0
    for r in agent.events:
        reward += agent.rewards[events[r]]
        if events[r] == 'COIN_COLLECTED':
            agent.coins += 1
    # save reward
    agent.reward = reward

    # destinction between training an test
    if agent.isTraining:
        agent.totalReward += agent.reward
    else:
        agent.testReward  += agent.reward

def end_of_episode(agent):
    """
    called after each episode. Rewards must to be evaluated here to take the last step into consideration.
    """
    if agent.isTraining:

        # count reward
        reward = 0.0
        for r in agent.events:
            reward += agent.rewards[events[r]]
            if events[r] == 'COIN_COLLECTED':
                agent.coins += 1

        # if all coin collected give extra reward
        if agent.coins==9:
            reward += 300

        # save reward
        agent.reward = reward
        agent.tree.memory.addSample((agent.preState, agent.actions.index(agent.preAction), agent.reward, None))
        agent.totalReward += agent.reward

        # every 100 runs train Tree
        runs = 10
        if agent.episode%runs==0:
            print('Epsilon:\t ', agent.tree.epsilon)
            print('Total Reward:\t ', agent.totalReward/float(runs))
            print('Coins:\t\t ', agent.coins/float(runs))

            # train
            agent.tree.replay()
            # save rewards
            saveToFile(agent)

            print('Trained Episodes ', agent.episode)
            # reset total reward for next 10 episodes
            agent.totalReward = 0
            agent.coins       = 0
            print('Samples: ', len(agent.tree.memory.samples))

        # pause training to test
        if agent.episode%1000==0:
            agent.isTraining = False
            print('\ntraining stopped')
            print('TEST')

        # update epsilon
        agent.tree.updateEpsilon(agent.episode, settings['n_rounds'])

        # next episode starts now
        agent.episode += 1

    else:
        # count test games
        agent.testEpisode += 1
        print(agent.testEpisode, agent.testGames)
        # if test at end continue training and save data to file
        if agent.testEpisode==agent.testGames:
            print('Total Reward:\t ', agent.testReward/100.)
            print('Coins:\t\t ', agent.coins/100.)
            # save test reward
            saveTestToFile(agent)
            # rest test
            agent.testEpisode = 0
            agent.testReward  = 0
            agent.coins       = 0
            print('training continued\n')
            # continue training
            agent.isTraining = True

    # add up global episode to determine if game is done
    agent.globalEpisode += 1

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

def saveToFile(agent):
    agent.file.write('{}\t{}\n'.format(agent.episode, agent.totalReward))
    agent.file.flush()

def loadFile(agent):
    agent.file = open(agent.fileName, 'w')

def loadTestFile(agent):
    agent.testFile = open(agent.testFileName, 'w')

def saveTestToFile(agent):
    agent.testFile.write('{}\t{}\n'.format(agent.episode-1, agent.testReward/agent.testGames))
    agent.testFile.flush()
