import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-1e-3,5,1)
y_inv = 1/x

rates = [1,10,20,30,40,50]
rates = [1,2,0.7,0.5]
ys = []
for r in rates:
    ys.append( np.exp(-x*r) )

#plt.plot(x,y_inv,label='inv')
for i,r in enumerate( rates ):
    plt.plot(x,ys[i],label='x*%s'%(r))
plt.legend()
plt.show()

