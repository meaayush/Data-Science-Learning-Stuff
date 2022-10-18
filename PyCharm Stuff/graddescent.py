# import numpy as np
# def gd(x, y, it):
#     m_curr = c_curr = 0
#     n = len(x)
#     learning_rate = 0.08
#     for i in range(it):
#         y_predicted = m_curr * x + c_curr
#         cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
#         md = -(2 / n) * sum(x * (y - y_predicted))
#         cd = -(2 / n) * sum((y - y_predicted))
#         m_curr = m_curr - (learning_rate * md)
#         c_curr = c_curr - (learning_rate * cd)
#         print(f"m: {m_curr} c: {c_curr} cost:{cost} iterations: {i}")
#
#
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([5, 7, 9, 11, 13])
# it = 10000
# gd(x, y, it)

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn import linear_model


# %matplotlib inline

def grad_desc(x, y, learning_rate, it):
    m_curr = c_curr = 0
    n = len(x)
    #     plt.plot(x,y,color='r',linewidth=5)
    cost_prev = 0
    for i in range(it):
        y_pred = m_curr * x + c_curr
        #         plt.plot(x,y_pred,color='greenyellow')
        cost = (1 / n) * sum([j ** 2 for j in (y - y_pred)])
        md = -(2 / n) * sum(x * (y - y_pred))
        cd = -(2 / n) * sum(y - y_pred)
        m_curr = m_curr - (learning_rate * md)
        c_curr = c_curr - (learning_rate * cd)
        if math.isclose(cost, cost_prev, rel_tol=1e-20):
            break
        cost_prev = cost
        print(f"m: {m_curr} c: {c_curr} cost: {cost} iterations: {i}")
    return m_curr, c_curr


def usingsk(x_sk, y_sk):
    alg = linear_model.LinearRegression()
    alg.fit(x_sk, y_sk)
    return alg.coef_, alg.intercept_


if __name__ == "__main__":
    df = pd.read_csv('test_scores.csv')
    x = df.math
    y = df.cs
    learning_rate = 0.0002

    m, c = grad_desc(x, y, learning_rate, 100000)
    print(f"m: {m} c: {c}")

    m_sk, c_sk = usingsk(df[['math']], y)
    print(f"m_sk: {m_sk} c_sk: {c_sk}")
