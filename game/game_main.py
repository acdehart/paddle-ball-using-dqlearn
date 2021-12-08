from time import time

import matplotlib
from matplotlib import pyplot as plt

from ball_game import ball_game
from agent import paddle_boy


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def main():
    # INITIALIZE ENVIRONMENT / MEASURE BASELINE
    SetupStage()
    scores = [0]*10
    lap = 0

    agent = paddle_boy()
    trial = ball_game(agent, show=True)  # Score is ready

    while True:
        lap += 1
        trial = ball_game(agent, show=True)  # Score is ready

        scores.pop(0)
        scores.append(agent.score)
        rolling_average = sum(scores)/len(scores)
        print(f"{lap}| Score {round(rolling_average, 2)} | {round(agent.score,2)}")

        # ax1.cla()
        # ax1.plot(scores)
        # ax1.plot([0, 10], [rolling_average, rolling_average], 'r--', alpha=0.3)
        # plt.show(block=False)


def SetupStage():
    fig = plt.figure()
    fig.set_size_inches(2, 1)
    ax1 = fig.add_subplot(1, 1, 1)
    plt.ylim(ymin=-1, ymax=50)
    plt.xlim(xmin=0, xmax=10)


def calculate_score(agent):
    hits = agent.hits
    run_time = time() - agent.start_time
    dist_bias = - abs((agent.ball_distance_x / 400) ** 2)
    print(f"H {hits} | RT {round(run_time, 1)}s | DB {round(dist_bias, 1)} | Score: {round(agent.score, 3)} ")
    score = hits ** 2 + run_time ** 2 + dist_bias
    return score


if __name__ == '__main__':
    main()
