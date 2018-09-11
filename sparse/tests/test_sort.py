import numpy as np

data = [0.9, 0.1, 0.5]
colnames = np.array([1, 4, 7])

sorted_order = list(np.argsort(data))
index_insorted = [sorted_order.index(i) for i in range(len(data))]
new_colnames = colnames[index_insorted]


print "ix in sorted", ix_insorted
print "new names", colnames[index_insorted]


