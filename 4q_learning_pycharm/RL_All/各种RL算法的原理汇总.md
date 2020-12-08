# 一、伪代码与原理总结

## 1.Q_learning

Q_learning是一种比较大胆的算法，它会敢于尝试损失很大的动作，更新较快，效果不错。

![Q_learning](D:\typora学习\插入图片保存位置\image-20201116103829773.png)



## 2.Sarsa

Sarsa是一种更为保守的算法，它会尽力保证不受到大的惩罚，但是很容易陷入决策的循环中，‘犹豫不决’。

![image-20201116104015627](D:\typora学习\插入图片保存位置\image-20201116104015627.png)

## 3.Sarsa($$\lambda$$)

在叙述Sarsa($$\lambda$$)之前，先来看一下单步更新和回合更新的区别。

单步更新：只会去更新最接近宝藏的步。

回合更新：得到宝藏路径上的所有步都会被认为是获得宝藏的必要条件，进行更新。

![image-20201116104827872](D:\typora学习\插入图片保存位置\image-20201116104827872.png)

但是，回合更新的坏处在于，如果在获得宝藏之前一直在原地打转，那么这些无效的步也会被更新。

![image-20201116105154886](D:\typora学习\插入图片保存位置\image-20201116105154886.png)

Sarsa($$\lambda$$)就是用来解决这个回合更新和单步更新的问题的。

**$$\lambda$$是一个（0,1）之间的值，**

它越小，代表对远离宝藏的步更新的偏袒程度越小，即对接近宝藏的步更新力度越大，远离的步更新力度越小。当$$\lambda$$ = 0，就是单步更新

它越大，代表对远离宝藏的步更新的偏袒程度越大，即对接近宝藏的更新力度稍大，远离的步的更新力度也不很小。当$$\lambda$$ = 1，就是回合更新。

![image-20201116105241746](D:\typora学习\插入图片保存位置\image-20201116105241746.png)



**算法伪代码：**

![image-20201116105732331](D:\typora学习\插入图片保存位置\image-20201116105732331.png)

![image-20201116105743099](D:\typora学习\插入图片保存位置\image-20201116105743099.png)

# 二、用代码实现它

## 1.互动环境的构造(env.py)

![image-20201116182841388](D:\typora学习\插入图片保存位置\image-20201116182841388.png)



![image-20201116182718403](D:\typora学习\插入图片保存位置\image-20201116182718403.png)

奖励的有关设置都在step函数里。

这个环境的**状态s**就是**红色方块的坐标observation**（左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标）

step函数接受一个动作（0,1,2,3），返回s_, reward, done。



main函数的作用是什么呢？是执行本程序并返回本程序里面的返回值，而不返回import的那些程序的返回或者打印值。

```python
##state就是每次探索者的四维坐标
import numpy as np
import time
import tkinter as tk

UNIT = 40     #每个格子的像素
MAZE_H = 4    #高四个格子
MAZE_W = 4    #宽四个格子

class Maze(tk.Tk, object): #继承tk
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'r', 'l'] 
        self.n_actions = len(self.action_space)  #actions的个数，4
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H*UNIT, MAZE_W*UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width = MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)


        # 起点
        origin = np.array([20, 20])

        # 黑洞
        # canvas里面的坐标是四维的，因为用四条线才能切出来一块形状（或者是圆的外切矩形），所以四维坐标是[左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标]
        ############################》x
        #     |    |
        #     |    |
        #-----######-----------------
        #     #    #
        #-----######---------------
        #     |    |
        #     |    |
        #     |    |
        #############################
       #y
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # 宝藏是一个椭圆
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # 红色方块
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        #重置方块，回到一开始的位置
        self.update()  #这个.update是tk自带的
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20,20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        #coords是返回一个目标目前的坐标。因为是一个矩形，[5,5,35,35]表示了左上角和右下角两个角的坐标
        #以这种四维的坐标代表state
        s = self.canvas.coords(self.rect)

        base_action = np.array([0,0])  #定位的二维列表

        if action == 0:   #根据动作列表的索引来指挥，这是up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])  #移动方块到目前的这个地址

        s_ = self.canvas.coords(self.rect)


        ##奖励函数
        #如果探索矩形坐标和宝藏椭圆重合，就结束，并且reward=1
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        #如果探索矩形和任何一个黑洞坐标重合，就扣分，并且结束游戏
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            s_ = 'terminal'
        #如果出现在空白的地方，则不计分
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        '渲染'
        time.sleep(0.1)
        self.update()  #self.update是tkinter自带的更新方法

def reupdate():         #并且这里不能传入因为后面用到了env.after传入
    '''传入的env是实例（类对应的实例'''
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, reupdate)
    env.mainloop()
```



## 2.RL算法的构造(RLearn_brain.py)

根据伪代码，这部分主要编写的是**循环里面的小步骤**

### 1)公用的类RL

在RL算法里面， 有几个功能是公用的:

* 选择动作，都是选q表里这个状态这一行里面最大值对应的动作
* 初始化传入值和常量，包括传入动作，学习率，奖励衰减率，贪婪系数，并且**初始化q表**。
* 检查这个状态（也就是红方块的坐标）是否出现过的函数，没出现就加到q表里

因为这些是相同的，所以写在一个类里，之后只需要继承他即可。没出现过就全0加入Q表里。

```python
import numpy as np
import pandas as pd


class RL:
    '''公用的主类，包括选择动作的方法，检查状态是否存在的方法。这些方法都是公用的'''
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) #建立一个空的pandas列表，表头是actions

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def learn(self, *args):
        '''这个方法需要别的类来编写'''
        pass

```

### 2)经典Qlearning类

经典的Q-learning算法按照下图编写新的类包含继承和learn方法即可。

<img src="D:\typora学习\插入图片保存位置\image-20201116192050823.png" alt="image-20201116192050823" style="zoom:67%;" />

```python
class QLearningTable(RL):
    '''继承RL类'''
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
```

### 3)Sarsa类

Sarsa算法也是编写一个新类并继承RL并编写learn算法即可，按照下图编写learn方法。

<img src="D:\typora学习\插入图片保存位置\image-20201116192118130.png" alt="image-20201116192118130" style="zoom:67%;" />

```python
class SarsaTable(RL):
    '''继承RL类'''
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        '''Sarsa的算法和qlearning的本质区别在于更新方式'''
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # 不用max了，就直接选择s_ , a_对应的值
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
```

### 4)SarsaLambda类

这个算法中，**check_state_exist**函数和RL公用类中的不太一样，因为除了要给Qtable添加未遇见过的state，还要给Etable添加，所以需要重写父类里面的这个函数！

E(s,a)是一个和Qtable类似的表，代码中用eligibility_trace表示

<img src="D:\typora学习\插入图片保存位置\image-20201116192227367.png" alt="image-20201116192227367" style="zoom:67%;" />

算法中的:
$$
E(S,A)\gets E(S,A)+1
$$
其实有两种方案可供选择：

![image-20201116185310154](D:\typora学习\插入图片保存位置\image-20201116185310154.png)

算法中的公式对应的是第二行的图，但是这样会带来一些麻烦，不如第三个图，每次遇到一个动作，在E表里面的值加到封顶就可以了。具体的实现下面代码里体现了。

```python
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay   # 这一行为啥不能直接用lambda，必须用lambda_，因为lambda是个内置函数！
        self.eligibility_trace = self.q_table.copy()   # 这个eligibility_trace用来复制Q表，完成一份Q表的copy,就是伪代码里面的E(S,A)

    def check_state_exist(self, state):
        '''这项和RL类里面不一样，需要重写父类的方法'''
        if state not in self.q_table.index:
            # append new state to q table
            to_be_appeded = pd.Series(
                [0]*len(self.actions),
                index=self.q_table.columns,
                name=state
                )
            self.q_table = self.q_table.append(to_be_appeded)

            # 同样也给eligibility_trace更新
            self.eligibility_trace = self.eligibility_trace.append(to_be_appeded)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s!= 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # 不用max了，就直接选择s_ , a_对应的值
        else:
            q_target = r  # next state is terminal
        # 从这里开始有了变化
        error = q_target - q_predict

        # 更新E(S,A)有两种模式
        # # 模式1：直接给表里对应的值+1，不封顶
        self.eligibility_trace.loc[s, a] += 1

        # 模式2：
        # self.eligibility_trace.loc[s, :] *= 0   # 和+=一样，不过是乘法而已,把这一行全变成0
        # self.eligibility_trace.loc[s, a] += 1   # 只给执行动作这里变成1

        # 更新Q表
        self.q_table += self.lr * error * self.eligibility_trace

        # 衰减eligibility_trace在更新之后
        self.eligibility_trace *= self.lambda_ * self.gamma
```



## 3.主循环函数的编写

### 1)run_this_Qlearning.py

Initialize Q(s,a) arbitrary这一步在main函数的RL这一步。

<img src="D:\typora学习\插入图片保存位置\image-20201116192311542.png" alt="image-20201116192311542" style="zoom:67%;" />

```python
#这个程序得到的表格记录了所有的状态，但是进行动作的时候，q表不是连续的，是来回跳的，路径的坐标是连续的。但是表里面每一行之间不连续，是来回跳的、
#但是选择关系仍然是根据目前表格中最大的选择
from env import Maze
from RLearn_brain import QLearningTable

def finalupdate():
    for episode in range(100):  #外循环
        # initial observation
        observation = env.reset()  # 探索者的位置observation，也就是s，每次循环之后就会更新

        while True:     #内循环
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
                
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))   #RL在这呢！！！这里也就初始了q表

    env.after(100, finalupdate)   # 刚开始让你看看地图，100ms不动之后，小球开始运动！
    env.mainloop()
```

### 2)run_this_Sarsa.py

Initialize Q(s,a) arbitrary这一步在main函数的RL这一步。

<img src="D:\typora学习\插入图片保存位置\image-20201116192324548.png" alt="image-20201116192324548" style="zoom:67%;" />

```python
#这个程序得到的表格记录了所有的状态，但是进行动作的时候，q表不是连续的，是来回跳的，路径的坐标是连续的。但是表里面每一行之间不连续，是来回跳的、
#但是选择关系仍然是根据目前表格中最大的选择、
from env import Maze
from RLearn_brain import SarsaTable


def finalupdate():
    for episode in range(100):
        # initial observation
        observation = env.reset()  # 探索者的位置observation，也就是s，每次循环之后就会更新

        #一开始就选一个动作
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # 根据action，得到目前的位置，奖励值，完成与否
            observation_, reward, done = env.step(action)

            # Sarsa将会立即选择动作。
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)   #RL中的learn函数也要改，要加上立即选择的动作，

            # swap observation，Sarsa就直接选取这个动作作为下一步的action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))   #RL在这呢！！！这里也就初始了q表

    env.after(100, finalupdate)
    env.mainloop()
```

### 3)run_this_Sarsalambda.py

需要每次外循环的第一行对E表清零。也就是小球落入陷阱或者走到宝藏就对E清零。

**其他和Sarsa一样**。这个伪代码里面写的是InitializeS，A，其实A的初始化就相当于Sarsa里面的上来就选一个A.

<img src="D:\typora学习\插入图片保存位置\image-20201116201326780.png" alt="image-20201116201326780" style="zoom:50%;" />

<img src="D:\typora学习\插入图片保存位置\image-20201116192512344.png" alt="image-20201116192512344" style="zoom: 80%;" />

```python
#这个程序得到的表格记录了所有的状态，但是进行动作的时候，q表不是连续的，是来回跳的，路径的坐标是连续的。但是表里面每一行之间不连续，是来回跳的、
#但是选择关系仍然是根据目前表格中最大的选择、

def finalupdate():
    for episode in range(100):
        
        # 使E表清零
        for i in range(len(RL.eligibility_trace.index)):
            RL.eligibility_trace.iloc[i, :] = 0

        # initial observation
        observation = env.reset()  # 探索者的位置，每次循环之后就会更新

        #一开始就选一个动作
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()
            
            # 根据action，得到目前的位置，奖励值，完成与否
            observation_, reward, done = env.step(action)

            # Sarsa将会立即选择动作。
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)   #RL中的learn函数也要改，要加上立即选择的动作，

            # swap observation，Sarsa就直接选取这个动作作为下一步的action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))   #RL在这呢！！！

    env.after(100, finalupdate)
    env.mainloop()
```

