import numpy as np
import matplotlib.pyplot as plt
import gym

env=gym.make('MountainCar-v0')

na=env.action_space.n
nx=env.observation_space.high.size

nrbf=16*np.ones(nx).astype(int)
width=1./(nrbf-1.)
sigma=width[0]/2.
den=2*sigma**2

xrang=np.zeros((2,nx))
xrang[0,:]=env.observation_space.low
xrang[1,:]=env.observation_space.high

n=np.prod(nrbf)#length of rbf 
fs=np.zeros(n)
theta=np.zeros((n,na))

c=np.zeros((n,nx))
for i in range(nrbf[0]):
    for j in range(nrbf[1]):
        c[i*nrbf[1]+j,:]=(i*width[1],j*width[0])


def normalize(s):
    y=np.zeros(len(s))
    for i in range(len(s)):
        y[i]=(s[i]-xrang[0,i])/(xrang[1,i]-xrang[0,i])
    return y

def phi(s):
    fs=np.zeros(n)
    for i in range(n):
        fs[i]=np.exp(-np.linalg.norm(s-c[i,:])**2/den)
    return fs

divide=200
x=np.linspace(xrang[0,0],xrang[1,0],num=divide)
xdot=np.linspace(xrang[1,1],xrang[0,1],num=divide)

q=np.zeros((divide,divide))
theta=np.load('theta.npy')
for i in range(divide):
    for j in range(divide):
        norm=normalize([x[j],xdot[i]])
        fs=phi(norm)
        q[i,j]=np.max(np.dot(theta.T,fs))

plt.imshow(q)
plt.colorbar()
plt.show()


np.save('q_s2000_reso200.npy',q)
