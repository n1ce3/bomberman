import numpy      as np
import matplotlib.pyplot as plt

# reward file
fileRewardName = 'eGreedyTest300MB.txt'
fileRewardNameTest = 'eGreedy300MB.txt'

def plotTotalReward(fileName, title):
    # load file in numpy arrays
    data = np.loadtxt(fileName, delimiter='\t')
    # plot data
    plt.figure()
    plt.title(title)
    plt.ylabel('Total Reward')
    plt.xlabel('Number of episodes')
    plt.plot(data[:,0], data[:,1])
    plt.show()


plotTotalReward(fileRewardName, 'Total reward over numer of trained episodes using e-greedy, 300 n')
plotTotalReward(fileRewardNameTest, 'Total reward over numer of trained episodes using e-greedy, 300 n')
