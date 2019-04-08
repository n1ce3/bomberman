import numpy      as np
import tensorflow as tf
import random

from settings import e, events, settings
from .model   import Model, Memory

'''
Training of a agent using DNQ
'''

# setup game
def setup(agent):
    """
    Here we set up our Agent. This function is called before the first step of the game.
    """
    # possible actions
    agent.actions     = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    agent.action      = None
    agent.preAction   = None

    # state
    agent.stateSize   = 211
    agent.state       = None
    agent.preState    = None

    agent.episode     = 1

    # setup model
    setupModel(agent)
# setup model and everything for the training
def setupModel(agent):

    # trainings flag
    agent.isTraining  = True

    # track training
    agent.testAfter   = 1000
    agent.testGames   = 100
    # here we need to calculated how many training runs we're gonna have
    agent.numRuns     = settings['n_rounds']-(settings['n_rounds']/(agent.testAfter+agent.testGames))*agent.testGames

    # set model
    agent.numActions  = len(agent.actions)
    agent.model       = Model(agent.actions, agent.stateSize, agent.numRuns, agent.isTraining)
    # saver to save the trained model
    agent.saver       = tf.train.Saver()
    # start tensorflow session
    agent.session     = tf.Session()
    # initialize global variables
    agent.session.run(agent.model.varInit)
    # restore model (to play the game. If you run agent in trainingsmode 'model' gets loaded and sets initial network)
    # keep in mind that in trainingsmode (agent.isTraining=True) 'model' will be overwritten after each test session.
    agent.saver.restore(agent.session, 'model/model')

    # track training
    agent.testEpisode   = 0
    agent.globalEpisode = 1
    agent.coins         = 0
    agent.crates        = 0
    agent.suicides      = 0
    agent.kills         = 0
    agent.gotKilled     = 0
    agent.survived      = 0

    # handle own bombs
    agent.bombPos     = None
    agent.timer       = 0
    agent.preDanger   = None

    # rewards
    # these rewards will be accessed after each episode in end_of_episode
    agent.reward        = 0
    agent.totalReward   = 0
    agent.rewards       = {'MOVED_LEFT':          -1,
                           'MOVED_RIGHT':         -1,
                           'MOVED_UP':            -1,
                           'MOVED_DOWN':          -1,
                           'WAITED':              -1,
                           'INTERRUPTED':         -1,
                           'INVALID_ACTION':     -15,
                           'BOMB_DROPPED':        -5,
                           'BOMB_EXPLODED':        0,
                           'CRATE_DESTROYED':     40,
                           'COIN_FOUND':           0,
                           'COIN_COLLECTED':     110,
                           'KILLED_OPPONENT':    700,
                           'KILLED_SELF':       -400,
                           'GOT_KILLED':        -400,
                           'OPPONENT_ELIMINATED':  0,
                           'SURVIVED_ROUND':     400}

    # save training
    if agent.isTraining:
        # name of files
        agent.fileName     = 'exp32.txt'
        agent.testFileName = 'exp32Test.txt'
        # create or overwrite existing files
        loadFile(agent)
        loadTestFile(agent)

# play game and receive rewards for training
def act(agent):
    """
    This function is called at each step of the game and determines what our Agent will do next.
    Therefore the state in which the agent is gets loaded. The corresponding action will be
    done regarding to the output of chooseAction, which passes the current State to a neural net.

    If game is in training mode the agent will do random actions due to explore the enviroment.
    The fraction of random moves is determined by epsilon.
    20% of the 'random' moves are not random but the agent will get a hint to move to the next
    coin since this improves convergence.
    """

    # call current state
    agent.state       = getCurrentState(agent)
    # decide which action to do
    agent.action      = agent.model.chooseAction(agent.state, agent.session, agent.game_state)
    # pass action to the game
    agent.next_action = agent.action

    # save pos to know later where my own bomb has been droped
    if (agent.action=='BOMB') & (agent.game_state['self'][3]==1):
        pos = agent.game_state['self']
        agent.bombPos = (pos[0],pos[1])
        agent.timer = 0
    else:
        # count to know if bomb is exploded
        agent.timer += 1
        if agent.timer >= 5:
            agent.bombPos = (0,0)

    # if training
    if agent.isTraining:
        # don't do this the first step since there is no previous state and no reward
        if agent.game_state['step'] > 1:
            agent.model.memory.addSample((agent.preState, agent.actions.index(agent.preAction), agent.reward, agent.state))

        # save for next step
        agent.preState  = agent.state
        agent.preAction = agent.action

def reward_update(agent):
    """
    This function is called before each move except the first one.
    Update rewards here after each move.
    The corresponding rewards are initialize in the setup methode.
    The only reward we need to define here is the reward the agent
    gets when is was able to collect all coins (which is almost
    never the case and a bit senseless)
    """
    # count reward
    reward = 0
    for r in agent.events:
        # if killed self dont punsih for got killed
        # but punish for got killed if not killed self
        if events[r] == 'GOT_KILLED':
            if list(agent.rewards.keys()).index('KILLED_SELF') in agent.events:
                pass
            else:
                agent.gotKilled += 1
        else:
            reward += agent.rewards[events[r]]

        # track progress
        if events[r] == 'COIN_COLLECTED':
            agent.coins += 1
        if events[r] == 'KILLED_SELF':
            agent.suicide += 1
        if events[r] == 'CRATE_DESTROYED':
            agent.crates += 1
        if events[r] == 'KILLED_OPPONENT':
            agent.kills += 1
        if events[r] == 'SURVIVED_ROUND':
            agent.survived += 1

    # save reward
    agent.reward       = reward
    agent.totalReward += agent.reward

def end_of_episode(agent):
    """
    This function is called at the end of each episode. After 100 episodes we do a replay
    to train the net. Since this function is called after the last last action we need
    to consider the rewards of the last action here.
    After 1000 episdoes the game runs in test mode over 100 episodes in which the agent makes
    his own decision. In the test mode the net will not be trained.
    """
    # reset time
    agent.timer = 0

    # reward of last episode
    # count reward
    reward = 0
    for r in agent.events:
        # if killed self dont punsih for got killed
        # but punsih for got killed if not killed self
        if events[r] == 'GOT_KILLED':
            if list(agent.rewards.keys()).index('KILLED_SELF') in agent.events:
                pass
            else:
                agent.gotKilled += 1
        else:
            reward += agent.rewards[events[r]]

        # track progress
        if events[r] == 'COIN_COLLECTED':
            agent.coins += 1
        if events[r] == 'KILLED_SELF':
            agent.suicides += 1
        if events[r] == 'CRATE_DESTROYED':
            agent.crates += 1
        if events[r] == 'KILLED_OPPONENT':
            agent.kills += 1
        if events[r] == 'SURVIVED_ROUND':
            agent.survived += 1

    # if all coins collected give extra reward
    if agent.coins==9:
        reward += 300

    # count total reward
    agent.reward       = reward
    agent.totalReward += agent.reward

    # if trainings mode
    if agent.isTraining:
        # save reward
        agent.model.memory.addSample((agent.preState, agent.actions.index(agent.preAction), agent.reward, None))


        # every 100 runs train model
        if agent.episode%100==0:
            # print results of last 100 episodes
            print('Epsilon:\t ',      agent.model.epsilon)
            print('Total Reward:\t ', agent.totalReward/100.)
            print('Coins:\t\t ',      agent.coins/100.)
            print('Crates:\t\t ',     agent.crates/100.)
            print('Suicides:\t ',     agent.suicides/100.)
            print('Kills:\t\t ',      agent.kills/100.)
            print('Got killed:\t ',   agent.gotKilled/100.)
            print('Survived:\t ',     agent.survived/100.)

            # train
            agent.model.replay(agent.session)
            print('Trained Episodes ', agent.episode-99, '-', agent.episode)

            # save rewards
            saveToFile(agent)

            # reset total reward for next 100 episodes
            agent.totalReward = 0
            agent.coins       = 0
            agent.crates      = 0
            agent.suicides    = 0
            agent.kills       = 0
            agent.gotKilled   = 0
            agent.survived    = 0

        # pause training to test
        if agent.episode%agent.testAfter==0:
            agent.isTraining = False
            print('\ntraining stopped')
            print('TEST')

        # update epsilon
        agent.model.updateEpsilon(agent.episode)

        # next episode starts now
        agent.episode += 1

    # if test mode
    else:
        # count test games
        agent.testEpisode += 1
        # if test at end continue training and save data to file
        if agent.testEpisode==agent.testGames:
            # print test result
            print('Total Reward:\t ', agent.totalReward/agent.testGames)
            print('Coins:\t\t ',      agent.coins/agent.testGames)
            print('Crates:\t\t ',     agent.crates/agent.testGames)
            print('Suicides:\t ',     agent.suicides/agent.testGames)
            print('Kills:\t\t ',      agent.kills/agent.testGames)
            print('Got killed:\t ',   agent.gotKilled/agent.testGames)
            print('Survived:\t ',     agent.survived/agent.testGames)

            # save test reward
            saveTestToFile(agent)
            # rest test
            agent.testEpisode = 0
            agent.totalReward = 0
            agent.coins       = 0
            agent.crates      = 0
            agent.suicides    = 0
            agent.kills       = 0
            agent.gotKilled   = 0
            agent.survived    = 0

            # continue training
            agent.isTraining  = True

            # save net after test
            agent.saver.save(agent.session, 'model')
            print('model saved!')

            print('training continued\n')

    # add up global episode to determine if game is done
    agent.globalEpisode += 1

# calculate game state
def getCurrentState(agent):
    """
    This function returns the current state as an numpy array. The particular
    definition of the state can be seen in the report.
    In short, we consider the arena (Walls, Crates and Empty), then we define
    a danger level for all bombs (between 0-1 depending on in how many steps the
    bomb will explode, is positiv for my bombs and negative for the opponents bombs),
    further we consider the coins which are collectable as well as we consider
    the enemies positions. Our agent occupies the middle of a 7x7 sub arena
    which is his view. But we also give it hints where and how far the nearest coin, crate
    or enemy is away.
    """
    # create numpy array of current state
    # get arena
    arena = agent.game_state['arena']
    # get coins
    coins = agent.game_state['coins']
    # get position
    pos   = agent.game_state['self']
    # get bombs
    bombs = agent.game_state['bombs']
    # get explosions
    expl  = agent.game_state['explosions']
    # get opponents positions
    opp   = [(o[0],o[1]) for o in agent.game_state['others']]

    # put two additional walls around the arena so that we can later get the subarena (7x7)
    size = arena.shape[0]
    area = np.full((size+4,size+4), -1.)
    # fill the inner part with original arena
    area[2:size+2,2:size+2] = arena.T

    # get arena for danger level
    danger            = area.copy()
    danger[danger==1] = 0
    # keep empty arena to know where walls are to determine where bombs will explode
    dangerEmpty       = danger.copy()

    # delete all walls since they are not relevant (information is in area)
    danger[danger==-1] = 0

    # place danger of bombs
    p = settings['bomb_power']
    for col,row,t in bombs:
        # calculate danger level
        dangerLevel = 1. - 0.17*(t+1)
        if agent.bombPos!=(col,row):
            # enemys bomb
            dangerLevel = -dangerLevel

        # danger level at bombs location
        danger[row+2,col+2] = dangerLevel

        # if two explosions will cross each other we will be more aware of the more dangerous one
        # check if wall above bomb
        for i in range(p):
            if dangerEmpty[row-i+1,col+2] == -1: break
            else:
                if abs(danger[row-i+1,col+2]) <= abs(dangerLevel):
                    danger[row-i+1,col+2] = dangerLevel
                else: pass

        # check if wall underneath
        for i in range(p):
            if dangerEmpty[row+3+i,col+2] == -1: break
            else:
                if abs(danger[row+3+i,col+2]) <= abs(dangerLevel):
                    danger[row+3+i,col+2] = dangerLevel
                else: pass

        # check if wall right
        for i in range(p):
            if dangerEmpty[row+2,col+3+i] == -1: break
            else:
                if abs(danger[row+2,col+3+i]) <= abs(dangerLevel):
                    danger[row+2,col+3+i] = dangerLevel
                else: pass

        # check if wall left
        for i in range(p):
            if dangerEmpty[row+2,col-i+1] == -1: break
            else:
                if abs(danger[row+2,col-i+1]) <= abs(dangerLevel):
                    danger[row+2,col-i+1] = dangerLevel
                else: pass

    # in first stept there is no preDanger
    # preDanger is used to know in this step if o bomb is mine or a enemies bomb (determined by the sign of dangerLevel)
    if agent.game_state['step']==1:
        agent.preDanger = np.zeros((size+4,size+4))


    # consider explosions of enemies
    danger[2:size+2,2:size+2][(expl.T != 0) & (agent.preDanger[2:size+2,2:size+2] < 0)] = -1
    # consider my explosions
    danger[2:size+2,2:size+2][(expl.T != 0) & (agent.preDanger[2:size+2,2:size+2] > 0)] = 1

    # save danger level
    agent.preDanger = danger

    # consider coins
    subCoins = np.zeros((7,7))
    for x,y in np.ndindex(subCoins.shape):
        coin = (pos[0]+2+y-3,pos[1]+2+x-3)
        if coin in coins:
            # consider in subarena
            subCoins[x,y] = 1
            # delete from coin list since it is already considert in subCoins
            index = coins.index(coin)
            coins.pop(index)

    # consider opponents
    subOpp = np.zeros((7,7))
    for x,y in np.ndindex(subOpp.shape):
        opponent = (pos[0]+2+y-3,pos[1]+2+x-3)
        if opponent in opp:
            # consider in subarena
            subOpp[x,y] = -1
            # delete from opp list since opponent is already considert in subOpp
            index = opp.index(opponent)
            opp.pop(index)
    # add my position
    subOpp[3,3] = 1

    ####### CREATE STATE ########
    # create current state
    currentState = []
    # position
    position = np.array([pos[0],pos[1]])
    # subarena boundaries
    b = [pos[1]+1-2, pos[1]+1+5, pos[0]+1-2, pos[0]+1+5]

    # add whether own bomb is possible or not
    # not neccessary since agent can learn it
    # currentState.append(pos[3])


    # consider nearest coin which is not in subArena
    # find nearest coin and distance
    directionCoin = getNearest(position, np.asarray(coins))
    # add to current state
    currentState += directionCoin


    # consider enemy who is not in subArena
    # find nearest opponent and distance
    directionOpp = getNearest(position, np.asarray(opp))
    # add to current state
    currentState += directionOpp


    # consider nearest crate which is not in subarena
    # kill all crates in subArena since they are already considert in subarena
    crates = area.copy()
    crates[b[0]:b[1],b[2]:b[3]] = np.zeros((7,7))
    # get all indices of crates
    idxC = np.argwhere(crates.T==1)
    # find nearest crate and distance
    directionCrate = getNearest(position, idxC)
    # add to current state
    currentState  += directionCrate


    # get arena around pos
    subArena  = area[b[0]:b[1],b[2]:b[3]].flatten()
    # get danger aroung pos
    subDanger = danger[b[0]:b[1],b[2]:b[3]].flatten()
    # get coins around pos
    subCoins  = subCoins.flatten()
    # get opponents around pos
    subOpp    = subOpp.flatten()

    currentState += [x for t in zip(subArena, subDanger, subCoins, subOpp) for x in t]

    return np.array(currentState)

def getNearest(pos, objs):
    '''
    This function returns the distance and the direction of the nearest object in
    objs to the agent at postion pos. The direction is more or less hot-one encoded.
    The distance is float between 0 and 1 normed by the maxdistance of the arena (sqrt(2)*15))
    '''
    code = []
    if objs.shape[0] != 0:
        dist = []
        for obj in objs:
            c = np.array([obj[0], obj[1]])
            a = np.array([pos[0], pos[1]])
            dist.append(np.linalg.norm(obj-pos))

        # nearest object
        nearest = objs[np.argmin(dist)]

        # calculate angle to agent clockwise
        ang   = np.arctan2(nearest[0]-pos[0], nearest[1]-pos[1])
        angle = np.rad2deg(-(ang - np.pi/2.) % (2 * np.pi))

        # search best direction to next coin
        # right
        if angle == 0:
            code += [1,0,0,0]
        # down
        elif angle == 90:
            code += [0,1,0,0]
        # left
        elif angle == 180:
            code += [0,0,1,0]
        # up
        elif angle == 270:
            code += [0,0,0,1]
        # down and right
        elif (0 < angle < 90):
            code += [1,1,0,0]
        # down and left
        elif (90 < angle < 180):
            code += [0,1,1,0]
        # up and left
        elif (180 < angle < 270):
            code += [0,0,1,1]
        # up and right
        elif (270 < angle < 360):
            code += [1,0,0,1]

    else:
        code += [0,0,0,0]
        dist  = [0]
    # norm dist to max distance possible in arena
    return [min(dist)/(np.sqrt(2)*14.)] + code

# save progress
def saveToFile(agent):
    agent.file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(agent.episode,
                                                               agent.totalReward/100.,
                                                               agent.coins/100.,
                                                               agent.crates/100.,
                                                               agent.suicides/100.,
                                                               agent.kills/100.,
                                                               agent.gotKilled/100.,
                                                               agent.survived/100.))
    agent.file.flush()

def loadFile(agent):
    agent.file = open(agent.fileName, 'w')

def loadTestFile(agent):
    agent.testFile = open(agent.testFileName, 'w')

def saveTestToFile(agent):
    agent.testFile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(agent.episode-1,
                                                                   agent.totalReward/agent.testGames,
                                                                   agent.coins/agent.testGames,
                                                                   agent.crates/agent.testGames,
                                                                   agent.suicides/agent.testGames,
                                                                   agent.kills/agent.testGames,
                                                                   agent.gotKilled/agent.testGames,
                                                                   agent.survived/agent.testGames))
    agent.testFile.flush()
