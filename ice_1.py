import numpy as np
import matplotlib.pyplot as plt
x= np.arange(0,10,1)
y=[1,3,2,5,7,8,8,9,10,12]
x_m= np.mean(x)
y_m =np.mean(y)
a=len(x)
x_sub=x-x_m
y_sub=y-y_m
b1_num_1=x_sub*y_sub
b1_num=sum(b1_num_1)
b1_den=sum(np.power(x_sub,2))
b1=b1_num/b1_den
print("b1 is",b1)
b2=y_m-(b1*x_m)
print("b2 is",b2)
plt.scatter(x,y,color='yellow')
plt.plot(x,(b1*x+b2),color='red')
plt.show()




