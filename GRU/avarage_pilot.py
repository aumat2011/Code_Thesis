import os, sys

import matplotlib.pyplot as plt
import numpy as np

rates = [0.5664, 0.5664, 0.5684, 0.5667, 0.5626]

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
plt.xlabel('Epoch', size=14)
plt.ylabel('Accuracy Rate', size=14)
plt.title('Overall Success Rate', size=16)
plt.legend()
plt.show()