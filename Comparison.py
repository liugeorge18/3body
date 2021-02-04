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

y1=[2.01,6,0] #m
y2=[-3,2,0] #m
y3=[5,-6,0] #m

z1=[2.001,6,0] #m
z2=[-3,2,0] #m
z3=[5,-6,0] #m

#Convert pos vectors to arrays
r1=np.array(r1,dtype="float64")
r2=np.array(r2,dtype="float64")
r3=np.array(r3,dtype="float64")

x1=np.array(x1,dtype="float64")
x2=np.array(x2,dtype="float64")
x3=np.array(x3,dtype="float64")

y1=np.array(y1,dtype="float64")
y2=np.array(y2,dtype="float64")
y3=np.array(y3,dtype="float64")

z1=np.array(z1,dtype="float64")
z2=np.array(z2,dtype="float64")
z3=np.array(z3,dtype="float64")

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

def ThreeBodyEquations3 (w,t,G,m1,m2,m3):
    y1=w[:3]
    y2=w[3:6]
    y3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    y12=np.linalg.norm(y2-y1)
    y13=np.linalg.norm(y3-y1)
    y23=np.linalg.norm(y3-y2)
    
    dv1bydt=K1*m2*(y2-y1)/y12**3+K1*m3*(y3-y1)/y13**3
    dv2bydt=K1*m1*(y1-y2)/y12**3+K1*m3*(y3-y2)/y23**3
    dv3bydt=K1*m1*(y1-y3)/y13**3+K1*m2*(y2-y3)/y23**3
    dy1bydt=K2*v1
    dy2bydt=K2*v2
    dy3bydt=K2*v3
    y12_derivs=np.concatenate((dy1bydt,dy2bydt))
    y_derivs=np.concatenate((y12_derivs,dy3bydt))
    v12_derivs=np.concatenate((dv1bydt,dv2bydt))
    v_derivs=np.concatenate((v12_derivs,dv3bydt))
    derivs3=np.concatenate((y_derivs,v_derivs))
    return derivs3

def ThreeBodyEquations4 (w,t,G,m1,m2,m3):
    z1=w[:3]
    z2=w[3:6]
    z3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    z12=np.linalg.norm(z2-z1)
    z13=np.linalg.norm(z3-z1)
    z23=np.linalg.norm(z3-z2)
    
    dv1bydt=K1*m2*(z2-z1)/z12**3+K1*m3*(z3-z1)/z13**3
    dv2bydt=K1*m1*(z1-z2)/z12**3+K1*m3*(z3-z2)/z23**3
    dv3bydt=K1*m1*(z1-z3)/z13**3+K1*m2*(z2-z3)/z23**3
    dz1bydt=K2*v1
    dz2bydt=K2*v2
    dz3bydt=K2*v3
    z12_derivs=np.concatenate((dz1bydt,dz2bydt))
    z_derivs=np.concatenate((z12_derivs,dz3bydt))
    v12_derivs=np.concatenate((dv1bydt,dv2bydt))
    v_derivs=np.concatenate((v12_derivs,dv3bydt))
    derivs4=np.concatenate((z_derivs,v_derivs))
    return derivs4

#Package initial parameters
init_params=np.array([r1,r2,r3,v1,v2,v3]) #Initial parameters
init_params=init_params.flatten() #Flatten to make 1D array
time_span=np.linspace(0,50,5000) #50 orbital periods and 5000 points

init_params2=np.array([x1,x2,x3,v1,v2,v3]) #Initial parameters
init_params2=init_params2.flatten() #Flatten to make 1D array

init_params3=np.array([y1,y2,y3,v1,v2,v3]) #Initial parameters
init_params3=init_params3.flatten() #Flatten to make 1D array

init_params4=np.array([z1,z2,z3,v1,v2,v3]) #Initial parameters
init_params4=init_params4.flatten() #Flatten to make 1D array

#Run the ODE solver
import scipy.integrate 
three_body_sol=sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2,m3))

three_body_sol2=sci.integrate.odeint(ThreeBodyEquations2,init_params2,time_span,args=(G,m1,m2,m3))

three_body_sol3=sci.integrate.odeint(ThreeBodyEquations3,init_params3,time_span,args=(G,m1,m2,m3))

three_body_sol4=sci.integrate.odeint(ThreeBodyEquations4,init_params4,time_span,args=(G,m1,m2,m3))

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


y1_sol=three_body_sol3[:,:3]
y2_sol=three_body_sol3[:,3:6]
y3_sol=three_body_sol3[:,6:9]

ycom_sol=(m1*y1_sol+m2*y2_sol+m3*y3_sol)/(m1+m2+m3)
y1com_sol=y1_sol-ycom_sol
y2com_sol=y2_sol-ycom_sol
y3com_sol=y3_sol-ycom_sol

v1_sol3= three_body_sol3[:,9:12]
v2_sol3= three_body_sol3[:,12:15]
v3_sol3= three_body_sol3[:,15:18]

vcom_sol3 = (m1*v1_sol3+m2*v2_sol3+m3*v3_sol3)/(m1+m2+m3)


z1_sol=three_body_sol4[:,:3]
z2_sol=three_body_sol4[:,3:6]
z3_sol=three_body_sol4[:,6:9]

zcom_sol=(m1*z1_sol+m2*z2_sol+m3*z3_sol)/(m1+m2+m3)
z1com_sol=z1_sol-zcom_sol
z2com_sol=z2_sol-zcom_sol
z3com_sol=z3_sol-zcom_sol

v1_sol4= three_body_sol4[:,9:12]
v2_sol4= three_body_sol4[:,12:15]
v3_sol4= three_body_sol4[:,15:18]

vcom_sol4 = (m1*v1_sol4+m2*v2_sol4+m3*v3_sol4)/(m1+m2+m3)

x1com_sol=x1com_sol-r1com_sol
x2com_sol=x1com_sol-r2com_sol
x3com_sol=x1com_sol-r3com_sol

y1com_sol=y1com_sol-r1com_sol
y2com_sol=y1com_sol-r2com_sol
y3com_sol=y1com_sol-r3com_sol

z1com_sol=z1com_sol-r1com_sol
z2com_sol=z1com_sol-r2com_sol
z3com_sol=z1com_sol-r3com_sol


#Create figure
fig=plt.figure(figsize=(12,10))

#Create 3D axes
ax=fig.add_subplot(111)

plt.grid()

#Plot the orbits
#ax.plot(time_span,r1com_sol[:,0],color="tab:red",linewidth=4, linestyle='-', label="Condition 0")
ax.plot(time_span,x1com_sol[:,0],color="purple",linewidth=4, linestyle='--', label="Condition 1")
ax.plot(time_span,y1com_sol[:,0],color="green",linewidth=4, linestyle='--', label="Condition 2")
ax.plot(time_span,z1com_sol[:,0],color="darkblue",linewidth=4, linestyle='--', label="Condition 3")

#Add a few more bells and whistles
ax.set_xlabel("time",fontsize=17)
ax.set_ylabel("Diference in x-position",fontsize=17)
ax.legend(loc="upper left",fontsize=17)