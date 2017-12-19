## Mountaincar by SARSA(lambda) with function approximation

based on: 

----------------------------------------
env: openai gym 'MountainCar-v0'

observation:

[0] position [-1.2,0.6]

[1] velocity [-0.07,0.07]

actions:

[0] push left

[1] no push

[2] push right

-----------------------------------------
setting: 

features: 16*16 RBF

episode length: 500 per trial

step length: 2000 per episode

-----------------------------------------

hyperparameters:

alpha=0.1

lambda=0.5

gamma=0.99

epsilon0=0.1

epsilonf=0.01

-----------------------------------------

results:

learning curve:
![alt test](https://github.com/ha5ha6/mtcar_sarsalambdaFA/blob/master/learning_curve.png)

Q:
![alt test](https://github.com/ha5ha6/mtcar_sarsalambdaFA/blob/master/q.png)
