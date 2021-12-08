from tkinter import *
import time
import random

import numpy as np

from game_objects import Ball, Paddle, Bricks
canvas = None


class ball_game:
    def __init__(self, agent, show=False):
        self.agent = agent
        root = Tk()
        root.title("Bounce")

        root.geometry(f"500x570+{self.get_window_displacement(show)}+0")
        root.resizable(0, 0)
        root.wm_attributes("-topmost", 1)
        canvas = Canvas(root, width=500, height=500, bd=0, highlightthickness=0, highlightbackground="Red", bg="Black")
        canvas.pack(padx=10, pady=10)
        score = Label(height=50, width=80, text="Score: 00", font="Consolas 14 bold")
        score.pack(side="left")
        root.update()

        score.configure(text="Score: 00")
        canvas.delete("all")
        BALL_COLOR = ["red", "yellow", "white"]
        BRICK_COLOR = ["PeachPuff3", "dark slate gray", "rosy brown", "light goldenrod yellow", "turquoise3", "salmon",
                       "light steel blue", "dark khaki", "pale violet red", "orchid", "tan", "MistyRose2",
                       "DodgerBlue4", "wheat2", "RosyBrown2", "bisque3", "DarkSeaGreen1"]
        random.shuffle(BALL_COLOR)
        self.paddle = Paddle(canvas, "blue", self.agent)

        bricks = self.Lay_Brick(BRICK_COLOR, canvas)

        self.ball = Ball(canvas, BALL_COLOR[0], self.paddle, bricks, score)
        root.update_idletasks()
        root.update()
        self.update_game_state()

        self.LoopGame(canvas, self.paddle, root)

        root.destroy()

    def get_window_displacement(self, show):
        if show:
            displacement = 0
        else:
            displacement = 3000
        return displacement

    def Lay_Brick(self, BRICK_COLOR, canvas):
        bricks = []
        for i in range(0, 5):
            b = []
            for j in range(0, 19):
                random.shuffle(BRICK_COLOR)
                tmp = Bricks(canvas, BRICK_COLOR[0])
                b.append(tmp)
            bricks.append(b)
        for i in range(0, 5):
            for j in range(0, 19):
                canvas.move(bricks[i][j].id, 25 * j, 25 * i)
        return bricks

    def update_game_state(self, action=(0, 1, 0)):
        self.agent.hits = self.ball.hit
        self.update_ball_distance(action)
        return self.agent.calculate_score()

    def update_ball_distance(self, action=(0, 1, 0)):
        ball_pos = self.ball.canvas.coords(self.ball.id)
        ball_x = (ball_pos[0] + ball_pos[2]) / 2
        ball_y = (ball_pos[1] + ball_pos[3]) / 3
        paddle_pos = self.paddle.canvas.coords(self.paddle.id)
        paddle_x = (paddle_pos[0] + paddle_pos[2]) / 2

        self.agent.ball_distance_x = paddle_x - ball_x
        self.agent.ball_distance_y = ball_y
        self.agent.left_distance = paddle_x
        self.agent.right_distance = 500-paddle_x

        if action[0]:
            self.agent.velocity -= action[0]*100
        if action[2]:
            self.agent.velocity += action[2]*100

    def LoopGame(self, canvas, paddle, root):
        # INITIALIZE
        frame_rate = 30
        epsilon = 0.1
        num_actions = 4
        batch_size = 100
        input_t = np.array([self.agent.ball_distance_x, self.agent.ball_distance_y,
                   self.agent.left_distance, self.agent.right_distance]).reshape(-1, 4)/500


        # for r in range(500):

        frame = 0
        while True:
            frame += 1
            # print(frame)
            self.agent.velocity = 0

            if paddle.pausec != 1:
                try:
                    canvas.delete(m)
                    del m
                except:
                    pass
                if not self.ball.bottom_hit:
                    # ACTION -> Explore?
                    actions = [0, 1, 0]
                    input_tm1 = input_t
                    if frame%frame_rate == 0:
                        baseline = self.agent.score
                        if np.random.rand() <= epsilon:
                            action = np.random.randint(0, num_actions, size=1)
                        else:
                            q = self.agent.model.predict(input_tm1)
                            action = np.argmax(q[0])
                            actions = [0]*3
                            actions[action] = q[0][action]
                    self.ball.draw()

                    # EVALUATE
                    reward = self.update_game_state(actions)  # Input Node or score?
                    paddle.draw()
                    root.update_idletasks()
                    root.update()

                    # TRAIN MODEL ON SAMPLES
                    input_t = np.array([self.agent.ball_distance_x, self.agent.ball_distance_y,
                                 self.agent.left_distance, self.agent.right_distance]).reshape(-1, 4)/500

                    if frame%frame_rate == 0:
                        print(f"A{action} | R{reward-baseline}")
                        self.agent.exp_replay.remember([input_tm1, action, reward-baseline, input_t], self.ball.bottom_hit)
                        inputs, targets = self.agent.exp_replay.get_batch(self.agent.model, batch_size=batch_size)
                        loss_i = self.agent.model.train_on_batch(inputs, targets)

                    time.sleep(0.001)
                    if self.ball.hit == 95:
                        canvas.create_text(250, 250, text="YOU WON !!", fill="yellow", font="Consolas 24 ")
                        root.update_idletasks()
                        root.update()
                        playing = False
                        break
                else:
                    canvas.create_text(250, 250, text="GAME OVER!!", fill="red", font="Consolas 24 ")
                    self.ball.hit = self.ball.hit
                    reward = -1
                    root.update_idletasks()
                    root.update()
                    playing = False
                    break
            else:
                try:
                    if m == None: pass
                except:
                    m = canvas.create_text(250, 250, text="PAUSE!!", fill="green", font="Consolas 24 ")
                root.update_idletasks()
                root.update()

