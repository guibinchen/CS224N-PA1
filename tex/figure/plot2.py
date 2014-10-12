data = """0.3367
0.3176
0.2946
0.2817
0.2745
0.2685
0.2649
0.2613
0.2597
0.2534"""

aer = [float(line) for line in data.split("\n")]

import matplotlib.pyplot as plt
plt.plot(range(10000, 110000, 10000), aer)
plt.xlabel('Size of Training Corpus')
plt.ylabel('AER')
plt.grid(True)
plt.legend(('AER vs. Size of Training Corpus'))
plt.savefig('aer.png', dpi=300)
