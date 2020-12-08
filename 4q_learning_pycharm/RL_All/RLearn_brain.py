"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

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
        # self.eligibility_trace.loc[s, a] += 1

        # 模式2：
        self.eligibility_trace.loc[s, :] *= 0   # 和+=一样，不过是乘法而已,把这一行全变成0
        self.eligibility_trace.loc[s, a] += 1   # 只给执行动作这里变成1

        # 更新Q表
        self.q_table += self.lr * error * self.eligibility_trace

        # 衰减eligibility_trace在更新之后
        self.eligibility_trace *= self.lambda_ * self.gamma

