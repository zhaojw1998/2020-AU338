import numpy as np
import time

np.set_printoptions(precision=3, suppress=True)

NUMBER_OF_SOURCE = 3

NUMBER_OF_LINK = 8

S_and_L = np.asarray(
    [[1, 1, 0, 1, 0, 0, 1, 0],
     [1, 0, 1, 0, 1, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 1, 1, 1]]
)

COLLISION_GRAPH = np.asarray(
    [[0, 1, 1, 1, 1, 0, 0, 0],
     [1, 0, 1, 1, 0, 0, 1, 0],
     [1, 1, 0, 0, 1, 0, 0, 0],
     [1, 1, 0, 0, 0, 1, 1, 1],
     [1, 0, 1, 0, 0, 0, 0, 1],
     [0, 0, 0, 1, 0, 0, 1, 1],
     [0, 1, 0, 1, 0, 1, 0, 1],
     [0, 0, 0, 1, 1, 1, 1, 0]], dtype=np.int32
)

# 表示冲突图


C_L = np.asarray([[10, 10, 10, 10, 10, 10, 10, 10]]).T  # 最大速率
DELAY_S = np.asarray([[10, 10, 10]]).T  # 时延限制

average_cl = np.zeros((NUMBER_OF_LINK, 1))
speed_s = np.ones((NUMBER_OF_SOURCE, 1))
sigma_l = np.ones((NUMBER_OF_LINK, 1))
p_l = np.ones((NUMBER_OF_LINK, 1))
mu_s = np.ones((NUMBER_OF_SOURCE, 1))

links_work_at_the_same_time = np.asarray(
    [
        [1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
    ], dtype=np.int32

)

k = 0
delta = 1
beta = 0.001
gama = 0.001

average_pl = np.zeros((NUMBER_OF_LINK, 1))
average_mu_s = np.zeros((NUMBER_OF_SOURCE, 1))
a = time.time()
while k < 20000:
    print("仿真步数：        ",k+1)
    print("每条链路的平均速率：", average_cl.T[0])  # 每条链路的平均速度
    print("p_{l}的平均值：   ", average_pl.T[0])
    print("mu_{s}的平均值：  ", average_mu_s.T[0])
    print("当前源速率:       ", speed_s.T[0])  # 源速率
    # print(sigma_l.T )# 每条链路的裕量
    # print(p_l.T )# 每条链路的拉格朗日乘子
    # print(mu_s .T)# 每个源的乘子

    print('\n\n')

    speed_s = 1 / S_and_L.dot(p_l)
    sigma_l = np.sqrt(S_and_L.T.dot(mu_s) / p_l)

    tmp = links_work_at_the_same_time.dot(p_l).sum(axis=1)

    work_mode = links_work_at_the_same_time[np.argmax(tmp)].reshape((NUMBER_OF_LINK, 1))

    k += delta
    average_cl = (average_cl * (k - 1) + delta * C_L * work_mode) / k

    p_l = p_l + beta * (S_and_L.T.dot(speed_s) - average_cl + sigma_l)

    mu_s = mu_s + gama * (S_and_L.dot(1 / sigma_l) - DELAY_S)

    p_l = np.maximum(p_l, 0.01)
    mu_s = np.maximum(mu_s, 0.01)
    average_pl = (average_pl * (k - 1) + p_l) / k
    average_mu_s = (average_mu_s * (k - 1) + mu_s) / k

print(str(time.time() - a))
