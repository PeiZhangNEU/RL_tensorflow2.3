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
        self.action_space = ['u', 'd', 'r', 'l'] #动作空间
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