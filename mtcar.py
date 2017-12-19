import gym
import numpy as np
import matplotlib.pyplot as plt

env=gym.make('MountainCar-v0')
outdir='experimentj/'
env=gym.wrappers.Monitor(env,outdir,video_callable=False,force=True)

na=env.action_space.n
nx=env.observation_space.high.size

nrbf=16*np.ones(nx).astype(int)
width=1./(nrbf-1.)
sigma=width[0]/2.
den=2*sigma**2

episodes=500
steps=2000
stpcount=0

epsilon0=0.1
epsilonf=0.01
epsilon_all=np.linspace(epsilon0,epsilonf,num=episodes)
#epsilonk=(epsilon-epsilonf)**(1./episodes)
lambd=0.5
alpha=0.1
gamma=0.99

xrang=np.zeros((2,nx))
xrang[0,:]=env.observation_space.low
xrang[1,:]=env.observation_space.high

n=np.prod(nrbf)#length of rbf 
fs=np.zeros(n)
fs_new=np.zeros(n)
theta=np.zeros((n,na))

#rbf centers
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

def egreedy(e,Q):
    rand=np.random.random()
    if rand<1.-e:
        a=Q.argmax()
    else:
        a=env.action_space.sample()
    return int(a)

def getQ(fs,theta):
    Q=np.dot(theta.T,fs)
    return Q

def getQact(fs,a,theta):
    Q=np.dot(theta[:,a],fs)


#data storage
qv_all=[]
delta_all=[]
r_all=[]
epsilon=epsilon0

for ep in range(episodes):

    #done=False
    r_sum=0
    epsilon=epsilon_all[ep]
    e=np.zeros((n,na))
    
    s=normalize(env.reset())
    fs=phi(s)
    Q_old=getQ(fs,theta)
    
    a=egreedy(epsilon,Q_old)
    
    for t in range(steps):

        
        s_new,r,done,info=env.step(a)
        s_new=normalize(s_new)
        fs_new=phi(s_new)
        Q=getQ(fs_new,theta)
        qv_all.append(Q)
        a_new=egreedy(epsilon,Q)
	
        if done:
            delta=r-Q_old[a]
        else:
            delta=r+gamma*Q[a_new]-Q_old[a]

        delta_all.append(delta)
        e[:,a]=fs #replace traces

        for a in range(n):
            for b in range(na):
                theta[a,b]+=alpha*delta*e[a,b]
	
        e*=gamma*lambd

        s=s_new
        fs=fs_new
        a=a_new
        Q_old=Q

        r_sum+=r
	
        if done:
            break

        stpcount+=1
        
    print("Ep:"+str(ep)+" t:"+str(t)+" t_all:"+str(stpcount)+" r:"+str(r_sum)+" epsilon:"+str(epsilon)+" Q:"+str(Q))
    #epsilon*=epsilonk
    r_all.append(r_sum)
    np.save('reward_j.npy',r_all)
    np.save('qvalue.npy',qv_all)
    np.save('delta.npy',delta_all)
    np.save('theta.npy',theta)

env.close()
    

    
