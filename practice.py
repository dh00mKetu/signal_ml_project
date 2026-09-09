import numpy as np


arr10=np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
arr10_square=arr10**2
arr10_sum=np.sum(arr10**2,axis=1)
print(arr10_sum)