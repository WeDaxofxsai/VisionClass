import matplotlib.pyplot as plt
import datetime
from pprint import pprint


def runin(_points):

    start = datetime.datetime.now()
    # Step 0
    n = len(_points) - 1
    points = []
    for i in _points:
        points.append(i[0])
        points.append(i[1])

    h = [points[(i + 1) * 2] - points[i * 2] for i in range(n)]
    A = [0] * n
    l = [0] * (n + 1)
    u = [0] * (n + 1)
    z = [0] * (n + 1)
    c = [0] * (n + 1)
    b = [0] * n
    d = [0] * n

    # Step 1
    for i in range(n):
        h[i] = points[(i + 1) * 2] - points[i * 2]

    # Step 2
    for i in range(1, n):
        A[i] = (
            3 * (points[(i + 1) * 2 + 1] - points[i * 2 + 1]) / h[i]
            - 3 * (points[i * 2 + 1] - points[(i - 1) * 2 + 1]) / h[i - 1]
        )

    # Step 3
    l[0] = 1
    u[0] = 0
    z[0] = 0

    # Step 4
    for i in range(1, n):
        l[i] = 2 * (points[(i + 1) * 2] - points[(i - 1) * 2]) - h[i - 1] * u[i - 1]
        u[i] = h[i] / l[i]
        z[i] = (A[i] - h[i - 1] * z[i - 1]) / l[i]

    # Step 5
    l[n] = 1
    z[n] = 0
    c[n] = 0

    # Step 6
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - u[j] * c[j + 1]
        b[j] = (points[(j + 1) * 2 + 1] - points[j * 2 + 1]) / h[j] - h[j] * (
            c[j + 1] + 2 * c[j]
        ) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    # for i in range(n):
    #     print(
    #         points[i * 2 + 1],
    #         "{:.16f}".format(b[i]),
    #         "{:.16f}".format(c[i]),
    #         "{:.16f}".format(d[i]),
    #         sep=" , ",
    #         end=" , \n",
    #     )

    pointsNum = 100
    pointList = []
    s_step = -1 if points[i * 2] > points[(i + 1) * 2] else 1
    for i in range(n):
        for x in range(
            points[i * 2] * pointsNum, points[(i + 1) * 2] * pointsNum, s_step
        ):
            y = (
                points[i * 2 + 1]
                + b[i] * (x / pointsNum - points[i * 2])
                + c[i] * (x / pointsNum - points[i * 2]) ** 2
                + d[i] * (x / pointsNum - points[i * 2]) ** 3
            )
            pointList.append((round(y, 2), round(x / pointsNum, 2)))

    # pprint(pointList)
    # 将点画出来
    plt.plot([p[0] for p in pointList], [p[1] for p in pointList])
    plt.show()


if __name__ == "__main__":
    for i in range(50, 10):
        print(i)
