import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import networkx as nx

INDIM=2
OUTDIM=10

def draw_graph(ax, C):
    G = nx.from_numpy_matrix(C)
    pos = {}
    for i in range(INDIM):
        pos[i] = (2+5*i, 1)
    for i in range(OUTDIM):
        pos[i+INDIM] = (i, 0)
    nx.draw(G, pos, ax=ax)

done = False
while not done:
    A = npr.choice(2, size=(OUTDIM, INDIM), p=(0.2, 0.8))
    done = np.min(np.sum(A, 1)) > 0

C = np.vstack((np.hstack((np.eye(INDIM), A.T)), np.hstack((A, np.eye(OUTDIM)))))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(174.0/25.4, 70/25.4))
plt.rcParams.update({'font.size': 6.0})
ax2.spy(C)
ax2.set_xticks(())
ax2.set_yticks(())
#ax2.axis("off")
draw_graph(ax1, C)
