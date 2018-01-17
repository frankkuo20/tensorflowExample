import matplotlib.pyplot as plt

plt.plot([i for i in range(50)], [i for i in range(50)])
plt.xlabel('step')
plt.ylabel('loss')
plt.show()
