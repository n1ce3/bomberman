
import numpy as np
import random
import json
from time import sleep

from settings import e, events, settings

def setup(agent):
    """
    Here we set up our Agent. This function is called before the first step of the game.
    """
    # possible actions
    agent.actions     = ['UP', 'DOWN', 'LEFT', 'WAIT', 'RIGHT', 'BOMB']

    # before the game load the Q-Tabel either for improving or for choosing actions
    agent.qPath       = "qTable.json"
    loadQTable(agent)



    # state
    agent.state       = None
    agent.preState    = None

    # action
    agent.action      = None
    agent.preAction   = None
    agent.preReward   = None

    # coins
    agent.nearestCoin = None

    # training
    agent.epsilon     = 0.4
    agent.eDecay      = 1-agent.epsilon/settings['n_rounds']
    agent.alpha       = 0.25
    agent.gamma       = 0.3
    agent.isTraining  = False
    agent.round       = 0

    # rewards
    agent.reward      = {'COIN_COLLECTED':100,
                         'MOVED_LEFT':-1,
                         'MOVED_RIGHT':-1,
                         'MOVED_UP':-1,
                         'MOVED_DOWN':-1,
                         'WAITED':0,
                         'INVALID_ACTION':-50,
                         'SEARCH_COIN':30,
                         'CRATE_DESTROYED':20,
                         'COIN_FOUND':40,
                         'BOMB_DROPPED':0}

def act(agent):
    """
    This function is called at each step of the game and determines what our Agent will do next.
    Therefore the state in which the agent is gets loaded. The corresponding action will be
    done regarding to the Q-Table. If game is in training mode the agent will do random actions
    due to explore the enviroment. The fraction of random moves is determined my epsilon.
    """
    agent.logger.INFO(agent.game_state['train'])
    # get state
    currentState(agent)

    # if state isn't in qTable: initialize
    # the initialization may be improved if necessary
    if agent.state not in agent.qTable:
        agent.qTable[agent.state] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0, 'WAIT':0, 'BOMB':0}

    # decide next move
    idx = np.argmax(list(agent.qTable[agent.state].values()))
    agent.next_action = agent.actions[idx]

    # while training explore environment
    if agent.isTraining:
        if random.random() < agent.epsilon:
            agent.next_action = random.choice(agent.actions)

    # save action
    agent.action = agent.next_action

def reward_update(agent):
    """
    This function is called after each move.
    """

    reward = 0
    for i in agent.events:
        # drop bomb should just give reward if crate destroyed
        if events[i] ==
        reward += agent.reward[events[i]]

    # reward for searching coin
    if (agent.action == agent.nearestCoin[1]):
        reward += agent.reward['SEARCH_COIN']

    # reward for dropping a bomb and find coin
    # 1. tried to drop bomb but invalid move
    #       is already considert in 'INVALID_ACTION' reward
    # 2. dropped a bomb couldnt find a coin
    #       wont give a penalty but may even be encouraged for a bit
    #       is already considert in 'CRATE_DESTROYED'
    # 3. dropped a bomb and revealed new coin
    #       is encouraged and already considert in 'COIN_FOUND'


    # dont do this the first steps since there is no previous state
    if agent.game_state['step'] > 2:
        # update Q-Table
        a  = agent.alpha
        g  = agent.gamma
        Q1 = agent.qTable[agent.preState][agent.preAction]
        Q2 = max(agent.qTable[agent.state].values())
        agent.qTable[agent.preState][agent.preAction] = Q1 + a*(agent.preReward + g*Q2 - Q1)

    else:
        agent.isTraining = True

    # safe action, state and gained reward
    agent.preAction = agent.action
    agent.preState  = agent.state
    agent.preReward = reward

def end_of_episode(agent):
    """
    """
    # print out number of trained episode
    agent.round += 1
    print('Round: ', agent.round, end='\r\r\r')
    agent.epsilon *= agent.eDecay

    # save Q-Table after every episode
    saveQTable(agent)

def currentState(agent):
    """
    This function gives the current state our agent is in. A state is defined as the state of the 8 tiles around
    the agent. On these tiles is either nothing, a wall or a coin (for the beginning).

    state = [upper, right, left, lower, nearestCoin, bombDroped, dangerLevel, dangerDirection]

    filled with w: wall
                c: coin
                e: empty
                g: crate
                x: explosion

                (u: nearest Coin is up
                d: nearest Coin is down
                l: nearest Coin is left
                r: nearest Coin is right)

                (b: bomb possible
                n: no bomb possible)

                (0: no danger
                1: explostion in three steps
                2: explostion in two steps
                3: explostion in one steps)

                (u: nearest bomb is up
                d: nearest bomb is down
                l: nearest bomb is left
                r: nearest bomb is right)
    """

    # get position
    pos       = agent.game_state['self']
    # get arena
    arena     = agent.game_state['arena']
    # get explosion
    explosion = agent.game_state['explosions']
    # get coins
    coins     = agent.game_state['coins']

    state = []

    moves = [(0, -1), (1, 0), (-1, 0), (0, 1)]

    # check the surounding four tiles
    for col, row in moves:
        tile = arena[pos[1]+row, pos[0]+col]
        # wall
        if tile == -1:
            state.append('w')
        # crate
        elif tile == 1:
            state.append('g')
        # empty, coin or explosion
        elif tile == 0:
            # if coin
            if (pos[0]+col, pos[1]+row) in coins:
                state.append('c')
            # if explosion
            elif explosion[pos[1]+row, pos[0]+col] != 0:
                state.append('x')
            # if empty
            else:
                state.append('e')

    # find the nearest coin
    agent.nearestCoin = findNearestCoin(pos, coins, arena)
    state.append(agent.nearestCoin[0])

    # check if own bomb is possible
    state.append('n') if pos[3] == 0 else state.append('b')



    agent.state = ''.join(state)

def findNearestCoin(pos, coins, arena):

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
            return 'd', 'DOWN'
        elif (arena[row][col+1]==-1) & (angle >= 315):
            return 'u', 'UP'
        else:
            return 'r', 'RIGHT'
    elif 45 <= angle < 135:
        if (arena[row+1][col]==-1) & (angle < 90):
            return 'r', 'RIGHT'
        elif (arena[row+1][col]==-1) & (angle >= 90):
            return 'l', 'LEFT'
        else:
            return 'd', 'DOWN'
    elif 135 <= angle < 225:
        if (arena[row][col-1]==-1) & (angle < 180):
            return 'd', 'DOWN'
        elif (arena[row][col-1]==-1) & (angle >= 180):
            return 'u', 'UP'
        else:
            return 'l', 'LEFT'
    elif 225 <= angle < 315:
        if (arena[row-1][col]==-1) & (angle < 270):
            return 'l', 'LEFT'
        elif (arena[row-1][col]==-1) & (angle >= 270):
            return 'r', 'RIGHT'
        else:
            return 'u', 'UP'
