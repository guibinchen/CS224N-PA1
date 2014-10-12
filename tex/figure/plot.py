data = """47.983  27.7
99.203  53.397
156.774 79.394
190.877 100.626
246.099 126.403
286.796 151.757
372.465 182.197
390.267 201.578
452.69  229.196
447.608 248.425"""

user = []
real = []
for line in data.split("\n"):
    ints = [float(x) for x in line.split()]
    user.append(ints[0])
    real.append(ints[1])

import matplotlib.pyplot as plt
plt.plot(range(10000, 110000, 10000), user, 'bs-', range(10000, 110000, 10000), real, 'r^-')
plt.xlabel('Size of Training Corpus')
plt.ylabel('Time (Seconds)')
plt.grid(True)
plt.legend(('Elapsed Time', 'Total CPU Time'), loc=4)
plt.savefig('time.png', dpi=300)
