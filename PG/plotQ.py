import matplotlib.pyplot as plt
import numpy as np
import pickle
import collections
import itertools

D = pickle.load(open('Q_his-lr-0.01.p','rb'))

deque_slice = collections.deque(itertools.islice(D, 0, 10001))
his = np.asarray(D)
s1 = np.hstack(his[:,0,2])
s2 = np.hstack(his[:,1,2])
s3 = np.hstack(his[:,2,2])
s4 = np.hstack(his[:,3,2])

plt.figure()
plt.plot(s1,label='Begin')
plt.plot(s2,label='s\'')
plt.plot(s3,label='s\'' )
plt.plot(s4,label= 's\'\'')
plt.legend(loc= 'best')
plt.show()