import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_pickle("result.pkl")
data.index = range(10, 51, 5)
#data.set_index(range(10, 51, 5))
print(data)


ax = data.T.plot.bar(rot=0)
ax.legend(loc=2, title='Window size')
plt.ylabel('Average Precision')
plt.show()