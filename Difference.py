#Import scipy and numpy
import scipy as sci
import numpy as np
#Import matplotlib and associated modules for 3D and animations
import matplotlib.pyplot as plt

#Define universal gravitation constant
G=6.67408e-11 #N-m2/kg2

#Reference quantities
m_nd=1.989e+30 #kg #mass of the sun
r_nd=1.496e+11 #m #astronomic unit
v_nd=30000 #m/s #relative velocity of earth around the sun
t_nd=365*24*3600 #s #orbital period of the Earth

#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

#Define masses
m1=1.1 #Body A
m2=0.9 #Body B
m3=1.0 #Body C

#Define initial position vectors
r1=[2,6,0] #m
r2=[-3,2,0] #m
r3=[5,-6,0] #m

x1=[2.02,6,0] #m
x2=[-3,2,0] #m
x3=[5,-6,0] #m


#Convert pos vectors to arrays
r1=np.array(r1,dtype="float64")
r2=np.array(r2,dtype="float64")
r3=np.array(r3,dtype="float64")

x1=np.array(x1,dtype="float64")
x2=np.array(x2,dtype="float64")
x3=np.array(x3,dtype="float64")

#Define initial velocities
v1=[-0.03,0.02,0] #m/s
v2=[-0.02,0.01,0] #m/s
v3=[0.04,-0.03,0] #m/s


#Convert velocity vectors to arrays
v1=np.array(v1,dtype="float64")
v2=np.array(v2,dtype="float64")
v3=np.array(v3,dtype="float64")

#A function defining the equations of motion 
def ThreeBodyEquations(w,t,G,m1,m2,m3):
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    r12=np.linalg.norm(r2-r1)
    r13=np.linalg.norm(r3-r1)
    r23=np.linalg.norm(r3-r2)
    
    dv1bydt=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    dv2bydt=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    dv3bydt=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    dr3bydt=K2*v3
    r12_derivs=np.concatenate((dr1bydt,dr2bydt))
    r_derivs=np.concatenate((r12_derivs,dr3bydt))
    v12_derivs=np.concatenate((dv1bydt,dv2bydt))
    v_derivs=np.concatenate((v12_derivs,dv3bydt))
    derivs=np.concatenate((r_derivs,v_derivs))
    return derivs

#A function defining the equations of motion 
def ThreeBodyEquations2 (w,t,G,m1,m2,m3):
    x1=w[:3]
    x2=w[3:6]
    x3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    x12=np.linalg.norm(x2-x1)
    x13=np.linalg.norm(x3-x1)
    x23=np.linalg.norm(x3-x2)
    
    dv1bydt=K1*m2*(x2-x1)/x12**3+K1*m3*(x3-x1)/x13**3
    dv2bydt=K1*m1*(x1-x2)/x12**3+K1*m3*(x3-x2)/x23**3
    dv3bydt=K1*m1*(x1-x3)/x13**3+K1*m2*(x2-x3)/x23**3
    dx1bydt=K2*v1
    dx2bydt=K2*v2
    dx3bydt=K2*v3
    x12_derivs=np.concatenate((dx1bydt,dx2bydt))
    x_derivs=np.concatenate((x12_derivs,dx3bydt))
    v12_derivs=np.concatenate((dv1bydt,dv2bydt))
    v_derivs=np.concatenate((v12_derivs,dv3bydt))
    derivs2=np.concatenate((x_derivs,v_derivs))
    return derivs2

#Package initial parameters
init_params=np.array([r1,r2,r3,v1,v2,v3]) #Initial parameters
init_params=init_params.flatten() #Flatten to make 1D array
time_span=np.linspace(0,50,5000) #50 orbital periods and 5000 points

init_params2=np.array([x1,x2,x3,v1,v2,v3]) #Initial parameters
init_params2=init_params2.flatten() #Flatten to make 1D array

#Run the ODE solver
import scipy.integrate 
three_body_sol=sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2,m3))

three_body_sol2=sci.integrate.odeint(ThreeBodyEquations2,init_params2,time_span,args=(G,m1,m2,m3))

r1_sol=three_body_sol[:,:3]
r2_sol=three_body_sol[:,3:6]
r3_sol=three_body_sol[:,6:9]

rcom_sol=(m1*r1_sol+m2*r2_sol+m3*r3_sol)/(m1+m2+m3)
r1com_sol=r1_sol-rcom_sol
r2com_sol=r2_sol-rcom_sol
r3com_sol=r3_sol-rcom_sol

v1_sol= three_body_sol[:,9:12]
v2_sol= three_body_sol[:,12:15]
v3_sol= three_body_sol[:,15:18]

vcom_sol = (m1*v1_sol+m2*v2_sol+m3*v3_sol)/(m1+m2+m3)


x1_sol=three_body_sol2[:,:3]
x2_sol=three_body_sol2[:,3:6]
x3_sol=three_body_sol2[:,6:9]

xcom_sol=(m1*x1_sol+m2*x2_sol+m3*x3_sol)/(m1+m2+m3)
x1com_sol=x1_sol-xcom_sol
x2com_sol=x2_sol-xcom_sol
x3com_sol=x3_sol-xcom_sol

v1_sol2= three_body_sol2[:,9:12]
v2_sol2= three_body_sol2[:,12:15]
v3_sol2= three_body_sol2[:,15:18]

vcom_sol2 = (m1*v1_sol2+m2*v2_sol2+m3*v3_sol2)/(m1+m2+m3)


#print the final positions
print("body A, for initial cond 1 x=",r1com_sol[-1,0],"y=",r1com_sol[-1,1])
print("body B, for initial cond 1 x=",r2com_sol[-1,0],"y=",r2com_sol[-1,1])
print("body C, for initial cond 1 x=",r3com_sol[-1,0],"y=",r3com_sol[-1,1])
print(" ")
print("body A, for initial cond 2 x=",x1com_sol[-1,0],"y=",r1com_sol[-1,1])
print("body B, for initial cond 2 x=",x2com_sol[-1,0],"y=",r2com_sol[-1,1])
print("body C, for initial cond 2 x=",x3com_sol[-1,0],"y=",r3com_sol[-1,1])

#Create figure
fig=plt.figure(figsize=(12,10))

#Create 3D axes
ax=fig.add_subplot(111)

plt.grid()

#Plot the orbits
ax.plot(r1com_sol[:,0],r1com_sol[:,1],color="tab:red",linewidth=3, linestyle=':', label="Referencia")
#ax.plot(r2com_sol[:,0],r2com_sol[:,1],color="tab:red")
#ax.plot(r3com_sol[:,0],r3com_sol[:,1],color="green")

ax.plot(x1com_sol[:,0],x1com_sol[:,1],color="purple", linewidth=3, label="Variación del 1%")
#ax.plot(x2com_sol[:,0],x2com_sol[:,1],color="tab:red", linestyle='dashed')
#ax.plot(x3com_sol[:,0],x3com_sol[:,1],color="green", linestyle='dashed')

#Plot the final positions of the stars
ax.scatter(r1com_sol[-1,0],r1com_sol[-1,1],color="tab:red",marker="<",s=150)
#ax.scatter(r2com_sol[-1,0],r2com_sol[-1,1],color="tab:red",marker="o",s=50,label="Body B")
#ax.scatter(r3com_sol[-1,0],r3com_sol[-1,1],color="green",marker="o",s=50,label="Body C")

ax.scatter(x1com_sol[-1,0],x1com_sol[-1,1],color="purple",marker="X",s=150)
#ax.scatter(x2com_sol[-1,0],x2com_sol[-1,1],color="tab:red",marker="X",s=50,label="Body B")
#ax.scatter(x3com_sol[-1,0],x3com_sol[-1,1],color="green",marker="X",s=50,label="Body C")

#Add a few more bells and whistles
ax.set_xlabel("x-coordinate",fontsize=17)
ax.set_ylabel("y-coordinate",fontsize=17)
ax.legend(loc="upper left",fontsize=17)

plt.xlim(-10,10)
plt.ylim(-7,10)

