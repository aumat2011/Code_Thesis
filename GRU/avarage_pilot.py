import os, sys

import matplotlib.pyplot as plt
import numpy as np

#rates = [0.5664, 0.5664, 0.5684, 0.5667, 0.5626] #OVERALL GRU SCORES - BEST FOR EACH FOLD   
#rates = [0.5671, 0.5657, 0.5673, 0.5662, 0.5626] #OVERALL LSTM0901 SCORES - BEST FOR EACH FOLD
#rates = [0.5668, 0.5650, 0.5660, 0.5674, 0.5637] #OVERALL LSTM1001 SCORES - BEST FOR EACH FOLD 
#rates = [0.5495, 0.5495, 0.5508, 0.5490, 0.5430] #OVERALL L2 1M - BEST FOR EACH FOLD
#rates = [0.5523, 0.5498, 0.5532, 0.5504, 0.5460] #OVERALL DropOut 1M - BEST FOR EACH FOLD
#rates = [0.5421, 0.5425, 0.5430, 0.5400, 0.5363] #OVERALL L1 1M - BEST FOR EACH FOLD
#rates = [0.5565, 0.5548, 0.5575, 0.5524, 0.5524] #OVERALL Adam 1M - BEST FOR EACH FOLD
#rates = [0.5560, 0.5540, 0.5561, 0.5529, 0.5518] #OVERALL LSTM Adam 1M - BEST FOR EACH FOLD
#rates = [0.5369, 0.5351, 0.5324, 0.5379,0.5343] #OVERALL GRU Adam 500 K - BEST FOR EACH FOLD
#rates = [0.5355, 0.5360, 0.5323, 0.5371,0.5325] #OVERALL LSTM Adam 500 K - BEST FOR EACH FOLD
#rates = [0.5134, 0.5128, 0.5191, 0.5166, 0.5112] #OVERALL GRU Adam 300 K - BEST FOR EACH FOLD
#rates = [0.5127, 0.5125, 0.5130, 0.5124, 0.5075] #OVERALL LSTM Adam 300 K - BEST FOR EACH FOLD
#rates = [0.5682, 0.5663, 0.5673, 0.5684, 0.5699] #OVERALL GRU NAdam 1500 K - BEST FOR EACH FOLD
#rates = [0.2669, 0.2663, 0.2732, 0.2675, 0.2675] #OVERALL GRU Adagrad 1500 K - BEST FOR EACH FOLD
#rates = [0.1289, 0.1268, 0.1259, 0.1307, 0.1248] #OVERALL GRU Adadelta 1500 K - BEST FOR EACH FOLD
#rates = [0.5548, 0.5521, 0.5538, 0.5528, 0.5489] #OVERALL GRU RMSprop 1500 K - BEST FOR EACH FOLD
rates = [0.3926, 0.3912, 0.3940, 0.3955, 0.3876] #OVERALL GRU SGD 1500 K - BEST FOR EACH FOLD

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
plt.title('Overall Success GRU with SGD on 1500K Records', size=16)
plt.legend()
plt.show()