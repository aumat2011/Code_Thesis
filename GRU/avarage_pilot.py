import os, sys

import matplotlib.pyplot as plt
import numpy as np

#rates = [0.5664, 0.5664, 0.5684, 0.5667, 0.5626] #OVERALL GRU SCORES - BEST FOR EACH FOLD   
#rates = [0.5671, 0.5657, 0.5673, 0.5662, 0.5626] #OVERALL LSTM0901 SCORES - BEST FOR EACH FOLD
#rates = [0.5668, 0.5650, 0.5660, 0.5674, 0.5637] #OVERALL LSTM1001 SCORES - BEST FOR EACH FOLD 
#rates = [0.5495, 0.5495, 0.5508, 0.5490, 0.5430] #OVERALL L2 1M - BEST FOR EACH FOLD
#rates = [0.5523, 0.5498, 0.5532, 0.5504, 0.5460] #OVERALL DropOut 1M - BEST FOR EACH FOLD
#rates = [0.5421, 0.5425, 0.5430, 0.5400, 0.5363] #OVERALL L1 1M - BEST FOR EACH FOLD
rates = [0.5565, 0.5548, 0.5575, 0.5524, 0.5524] #OVERALL Adam 1M - BEST FOR EACH FOLD

def lrfn(epoch):
    return rates[epoch]

avg = sum(rates) / len(rates)

avgS = [avg, avg, avg, avg, avg]

rng = [i for i in range(5)]
y = [lrfn(x) for x in rng]

fig, ax = plt.subplots()
ax.plot(rng, y, '-o', label='Epoch Rates')

# Plot the average line
plt.plot(rng, avgS, '-o', label='Average')

# Add data labels for each point in the "Epoch Rates" line
for i, txt in enumerate(y):
    ax.text(rng[i], txt, f'{txt:.4f}', ha='right', va='bottom')

# Add data label for the last point in the "Average" line
ax.text(rng[-1], avgS[-1], f'{avgS[-1]:.4f}', ha='right', va='bottom')

plt.xticks(rng, [f'{val:.1f}' for val in rng])
plt.grid()
plt.xlabel('Fold', size=14)
plt.ylabel('Accuracy Rate', size=14)
plt.title('Overall Success GRU on 1M records', size=16)
plt.legend()
plt.show()