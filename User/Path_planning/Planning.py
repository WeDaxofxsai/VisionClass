import time
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import heapq
from pprint import pprint
from runin import runin

"""
提升空间
1. 障碍物扫描能够改，障碍物的类记住障碍物中心点，判断路径中的点和障碍物中心点距离两倍车长，然后修改代价
2. 修改地图大小，重置栅格化
"""


car_width = 60  # 车的宽度
car_hight = 60  # 车的高度


def is_colision(x, y, map):
    # 障碍物的判断，这里可以根据实际情况进行修改
    if car_hight == 0 or car_width == 0:
        return True
    for ser_x in range(
        int(x - car_width / 2), int(x + car_width / 2), int(car_width / 10)
    ):
        for ser_y in range(
            int(y - car_hight / 2), int(y + car_hight / 2), int(car_hight / 10)
        ):
            if map.map[ser_x][ser_y].val == 1:
                return False
    return True


# 地图上的每一个点都是一个Point对象，用于记录该点的类别、代价等信息
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.val = 0  # 0代表可通行，1代表障碍物
        self.cost_g = 0  # 三个代价
        self.cost_h = 0
        self.cost_f = 0
        self.parent = None  # 父节点
        self.is_open = 0  # 0：不在开集合  1：在开集合  -1：在闭集合

    # 用于heapq小顶堆的比较
    def __lt__(self, other):
        return self.cost_f < other.cost_f


class Map:
    def __init__(self, map_size):
        self.map_size = map_size
        self.width = map_size[0]
        self.height = map_size[1]
        self.map = [
            [Point(x, y) for y in range(self.map_size[1])]
            for x in range(self.map_size[0])
        ]

    # 手动设置障碍物，可多次调用设置地图
    # 由于地图方向不同，这里的topleft并不总是左上角，topleft代表x和y全都较小的点
    def set_obstacle(self, topleft, width, height):
        for x in range(topleft[0], topleft[0] + width):
            for y in range(topleft[1], topleft[1] + height):
                self.map[x][y].val = 1


class AStar:
    def __init__(self, map, start_point, end_point, connect_num=8):
        self.map: Map = map
        self.start_point = start_point
        self.end_point = end_point
        self.open_set = [self.start_point]  # 开集合，先放入起点，从起点开始遍历
        self.w_x = 1.2  # 平衡代价参数
        self.w_y = 1.8
        self.start_point.is_open = 1  #
        self.connect_num = connect_num  # 连通数，目前支持4连通或8连通
        self.diffuse_dir = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ]  # 遍历的8个方向，只需取出元组，加到x和y上就可以

    def y_cost(self, p):
        """
        平衡代价，使得不同方向的代价不一样
        """
        x_dis = abs(self.end_point.x - p.x)
        y_dis = abs(self.end_point.y - p.y)

        return self.w_x * x_dis + self.w_y * y_dis

    def g_cost(self, p):
        """
        计算 g 代价，当前点与父节点的距离 + 父节点的 g 代价（欧氏距离）
        :param p: 当前扩散的节点
        :return: p 的 g 代价
        """
        x_dis = abs(p.parent.x - p.x)
        y_dis = abs(p.parent.y - p.y)
        return (
            np.sqrt(x_dis**2 + y_dis**2) + p.parent.cost_g
        )  # x_dis + y_dis + p.parent.cost_g  # np.sqrt(x_dis**2 + y_dis**2) + p.parent.cost_g

    def h_cost(self, p):
        """
        计算 h 代价，当前点与终点之间的距离（欧氏距离）
        :param p: 当前扩散的节点
        :return: p 的 h 代价
        """
        x_dis = abs(self.end_point.x - p.x)
        y_dis = abs(self.end_point.y - p.y)
        return np.sqrt(
            x_dis**2 + y_dis**2
        )  # x_dis + y_dis  # np.sqrt(x_dis**2 + y_dis**2)

    def is_valid_point(self, x, y):
        # 无效点：超出地图边界或为障碍物
        if x < 0 or x >= self.map.width:
            return False
        if y < 0 or y >= self.map.height:
            return False
        return self.map.map[x][y].val == 0 and is_colision(x, y, self.map)

    def search(self):
        self.start_time = time.time()  # 用于记录搜索时间
        p = self.start_point
        # p 为当前遍历节点，等于终点停下
        while not (p == self.end_point):
            # 弹出代价最小的开集合点，若开集合为空，说明没有路径
            try:
                p = heapq.heappop(self.open_set)
            except:
                raise "No path found, algorithm failed!!!"

            p.is_open = -1

            # 遍历周围点
            for i in range(self.connect_num):
                dir_x, dir_y = self.diffuse_dir[i]
                self.diffusion_point(p.x + dir_x, p.y + dir_y, p)
        return self.build_path(p)  # p = self.end_point

    def diffusion_point(self, x, y, parent):
        # 无效点或者在闭集合中，跳过
        if not self.is_valid_point(x, y) or self.map.map[x][y].is_open == -1:
            return
        p = self.map.map[x][y]
        pre_parent = p.parent
        p.parent = parent
        # 先计算出当前点的总代价
        cost_g = self.g_cost(p)
        cost_h = self.h_cost(p)
        cost_y = self.y_cost(p)
        cost_f = cost_g + cost_h + cost_y
        # 如果在开集合中，判断当前点和开集合中哪个点代价小，换成小的，相同x,y的点h值相同，g值不一定相同
        if p.is_open == 1:
            if cost_f < p.cost_f:
                # 如果从当前parent遍历过来的代价更小，替换成当前的代价和父节点
                p.cost_g, p.cost_h, p.cost_f = cost_g, cost_h, cost_f
            else:
                # 如果从之前父节点遍历过来的代价更小，保持之前的代价和父节点
                p.parent = pre_parent
        else:
            # 如果不在开集合中，说明之间没遍历过，直接加到开集合里就好
            p.cost_g, p.cost_h, p.cost_f = cost_g, cost_h, cost_f
            heapq.heappush(self.open_set, p)
            p.is_open = 1

    def build_path(self, p):
        print("search time: ", time.time() - self.start_time)
        # 回溯完整路径
        path = []
        while p != self.start_point:
            path.append(p)
            p = p.parent
        print("search time: ", time.time() - self.start_time)
        # 打印开集合、闭集合的数量
        print("open set count: ", len(self.open_set))
        close_count = 0
        for x in range(self.map.width):
            for y in range(self.map.height):
                close_count += 1 if self.map.map[x][y] == -1 else 0
        print("close set count: ", close_count)
        print("total count: ", close_count + len(self.open_set))
        path = path[::-1]  # path为终点到起点的顺序，可使用该语句翻转
        return path


if __name__ == "__main__":
    map_proportion = 1
    map = Map((int(500 / map_proportion), int(500 / map_proportion)))

    # 用于显示plt图

    ax = plt.gca()
    ax.set_xlim([0, map.width])
    ax.set_ylim([0, map.height])
    ax.set_aspect("equal")

    obstacle_list = [
        [
            [int(221 / map_proportion), int(210 / map_proportion)],
            int(58 / map_proportion),
            int(80 / map_proportion),
        ],
        [
            [int(218 / map_proportion), int(400 / map_proportion)],
            int(64 / map_proportion),
            int(60 / map_proportion),
        ],
        [
            [int(2 / map_proportion), int(200 / map_proportion)],
            int(22 / map_proportion),
            int(100 / map_proportion),
        ],
        [
            [int(476 / map_proportion), int(200 / map_proportion)],
            int(22 / map_proportion),
            int(100 / map_proportion),
        ],
        [
            [int(115 / map_proportion), int(235 / map_proportion)],
            int(30 / map_proportion),
            int(30 / map_proportion),
        ],
        [
            [int(355 / map_proportion), int(235 / map_proportion)],
            int(30 / map_proportion),
            int(30 / map_proportion),
        ],
        [[115, 355], 30, 30],
    ]
    for obstacle in obstacle_list:
        map.set_obstacle(obstacle[0], obstacle[1], obstacle[2])
        ax.add_patch(
            Rectangle(obstacle[0], width=obstacle[1], height=obstacle[2], color="gray")
        )

    cross_list = [
        [70, 70],
        [70, 190],
        [70, 310],
        [70, 430],
        [190, 70],
        [190, 190],
        [190, 310],
        [190, 430],
        [310, 70],
        [310, 190],
        [310, 310],
        [310, 430],
        [430, 70],
        [430, 190],
        [430, 310],
        [430, 430],
    ]
    for cross in cross_list:
        ax.plot(
            int(cross[0] / map_proportion),
            int(cross[1] / map_proportion),
            ".",
            color="b",
        )  # 'o' 表示圆形标记

    s_points = [
        [int(130 / map_proportion), int(35 / map_proportion)],
        [int(370 / map_proportion), int(35 / map_proportion)],
    ]  # 235, 210
    for p in s_points:
        ax.plot(p[0], p[1], ".", color="r")  # 'o' 表示圆形标记

    # 设置起始点和终点，并创建astar对象
    start_point = map.map[70][190]  # map.map[s_points[0][0]][s_points[0][1]]
    end_point = map.map[190][430]
    astar = AStar(map, start_point, end_point)
    path = astar.search()
    cnt = 25
    points = []
    for i in range(len(path)):
        ax.add_patch(Rectangle([path[i].x, path[i].y], width=1, height=1, color="red"))
        if i != 0 and path[i].y == path[i - 1].y:
            cnt = 0
        else:
            if cnt % 25 == 0 and cnt != 0:
                points.append([path[i].y, path[i].x])
                ax.plot(path[i].x, path[i].y, ".", color="g")  # 'o' 表示圆形标记
                cnt = 0
            cnt += 1

    # plt.savefig('./output/tmp.jpg')  # 可选择将其保存为本地图片
    points.append([path[-1].y, path[-1].x])
    # pprint(points)
    for point in points:
        print(point[0] * map_proportion - 35, ",", point[1] * map_proportion - 130, ",")
    print(len(points))
    runin(points)
    plt.show()
    time.sleep(1000)
