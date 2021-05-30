# Model checking for Bachelor's Thesis

#*.npy files are binary files to store numpy arrays. They are created with
# Example usage of npy files
import numpy as np

#data = np.random.normal(0, 1, 100)
#np.save('data.npy', data)
#And read in like

q_table = np.load('/Users/cedricsegers/Desktop/ComputerScience/3Bachelor/Bachelproef/qtables/87550-qtable.npy')

#print(q_table.shape)

np.save('Qtable87550.npy', q_table)
