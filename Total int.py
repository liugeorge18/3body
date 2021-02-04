import numpy as np
import matplotlib.pyplot as plt


#Convert pos vectors to arrays
r1=np.array([-3,0])
r2=np.array([5,3])
r3=np.array([10,0])


v12=np.array([4,3.4])
v13=np.array([5.8,0])
v21=np.array([-4,-3.4])
v23=np.array([3.33,-4.6])
v31=np.array([-5.8,0])
v32=np.array([-3.33,4.6])

v1 =np.add( v12 , v13)
v2 =np.add( v21 , v23)
v3 =np.add( v31 , v32)

fig, ax=plt.subplots(figsize=(12,8))

#Plot the initial positions 
ax.scatter(r1[:1],r1[1:2],color="darkblue",marker="o",s=100,label="Body A")
ax.scatter(r2[:1],r2[1:2],color="tab:red",marker="o",s=100,label="Body B")
ax.scatter(r3[:1],r3[1:2],color="green",marker="o",s=100,label="Body C")


plt.quiver(r1[0], r1[1], v1[0], v1[1], scale=50, width=0.005)
plt.quiver(r2[0], r2[1], v2[0], v2[1], scale=50, width=0.005)
plt.quiver(r3[0], r3[1], v3[0], v3[1], scale=50, width=0.005)


#Other stuffs
ax.set_xlabel("x-coordinate",fontsize=17)
ax.set_ylabel("y-coordinate",fontsize=17)
ax.legend(loc="upper left",fontsize=17)
plt.grid()

#Set limit to axis
plt.xlim([-8,13])
plt.ylim([-1.5,4.5])