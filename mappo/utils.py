import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.save(figure_file)

def obs_list_to_state_vector(obseravtion):
    '''
    flattening all the obs for all the agents
    '''
    state = []
    for row in obseravtion:
        obs = np.array([])
        for o in row:
            obs = np.concatenate([obs, o])
        state.append(obs)
    return np.array(state)