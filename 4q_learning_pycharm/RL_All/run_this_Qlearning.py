#这个程序得到的表格记录了所有的状态，但是进行动作的时候，q表不是连续的，是来回跳的，路径的坐标是连续的。但是表里面每一行之间不连续，是来回跳的、
#但是选择关系仍然是根据目前表格中最大的选择、

"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from env import Maze
from RLearn_brain import QLearningTable


def finalupdate():
    for episode in range(100):
        # initial observation
        observation = env.reset()  # 探索者的位置，每次循环之后就会更新
        print('the episod is:{}'.format(episode))
        i = 0

        while True:
            # fresh env
            env.render()
            print("the intro epoch is {}".format(i))

            print('the state now is {}'.format(observation))

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            print('the action is {}'.format(action))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            print('the state_pred is {}'.format(observation_))
            print('the reward is {}'.format(reward))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            print(RL.q_table)
            print('\n')

            # swap observation
            observation = observation_
            i += 1

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))   #RL在这呢！！！

    env.after(100, finalupdate)
    env.mainloop()