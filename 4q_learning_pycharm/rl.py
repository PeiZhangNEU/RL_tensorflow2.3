import numpy as np
import pandas as pd
import time

N_STATES = 6   # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 13   # 最大回合数
FRESH_TIME = 0.3    # 移动间隔时间

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table 全 0 初始
        columns=actions,    # columns 对应的是行为名称
    )
    return table

# q_table:
"""
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""


# 在某个 state 地点, 选择行为
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
        ##这个没问题，出来是一个字符‘left’或者是‘right’


    else:
        action_name = state_actions.argmax()  # 贪婪模式
        # 这里出大问题，出来之后不是字符，而是索引，所以根本起不到控制的作用了
        # 所以跟着下面用iloc也出错了！！！
        # 所以需要把这个索引转换成字符！！！
        demo_actions = ['left', 'right']
        action_name = demo_actions[action_name]

        # 这个样子，出来的 就也是字符了！！

    return action_name

def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate  #看啊，这个s=4的条件是在下一步动作为像右的基础上的，所以我理解的没有错！
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 2
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '-----T' our environment
    if S == 'terminal':
        print('oh!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('totle step = {}'.format(step_counter))
        time.sleep(2)
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
for episode in range(MAX_EPISODES):  # 回合
    step_counter = 0
    S = 0  # 回合初始位置
    is_terminated = False  # 是否回合结束

    # 外循环也需要更新一下环境
    update_env(S, episode, step_counter)  # 环境更新
    print('\n')

    epoch = 0
    while not is_terminated:

        print('the episode is{},and the epoch is{}'.format(episode, epoch))
        print(q_table)

        A = choose_action(S, q_table)  # 选行为
        print('now s is {}'.format(S))
        print('now action is {}'.format(A))

        S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
        print('this epoch ,the s_pred is {}'.format(S_))
        print('this epoch ,the reward is {}'.format(R))

        ############################################################################
        # 由于A是一个字符串'left'或者'right'而pandas现在删除了ix方法，所以需要使用iloc
        flag = 0
        if A == 'left':
            flag = 0
        else:
            flag = 1
        ##############################################################################

        q_predict = q_table.iloc[S, flag]  # 估算的(状态-行为)值
        if S_ != 'terminal':
            q_target = R + GAMMA * q_table.iloc[S_, :].max()  # 实际的(状态-行为)值 (回合没结束)
        else:
            q_target = R  # 实际的(状态-行为)值 (回合结束)
            is_terminated = True  # terminate this episode

        q_table.iloc[S, flag] += ALPHA * (q_target - q_predict)  # q_table 更新

        S = S_  # 探索者移动到下一个 state
        print('\n')

        step_counter += 1
        update_env(S, episode, step_counter)  # 环境更新

        print('\n')

        epoch += 1


