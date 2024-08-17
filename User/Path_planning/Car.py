import matplotlib.pyplot as plt
from pprint import pprint


class Car:
    def __init__(self, width=0, length=0, position_x=0, position_y=0, angle=90):
        self.width = width
        self.length = length
        self.n_position_x = position_x
        self.n_position_y = position_y
        self.n_angle = angle
        self.Vx = 0
        self.Vy = 0
        self.Vw = 0

    def car_move(self, x, y):
        pass

    def car_turn(self, angle):
        pass

    def get_position(self):
        return self.n_position_x, self.n_position_y

    def get_angle(self):
        return self.n_angle


class Map:
    def __init__(self, width, length):
        self.width = width
        self.length = length

    def is_valid_position(self, x, y):
        pass


# 定义事件处理器


if __name__ == "__main__":
    map = [[(x, y) for y in range(5)] for x in range(5)]

    pprint(map)
