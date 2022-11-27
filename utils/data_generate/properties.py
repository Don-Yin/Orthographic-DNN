from random import randint, uniform


def get_random_rgb(count: int):
    """
    :param count: number of rgb values
    """
    colors: set = set()
    while len(colors) < count:
        colors.update([(randint(0, 255), randint(0, 255), randint(0, 255))])
    return colors


def get_random_position(count: int):
    """
    :param count: number of positions
    """
    positions: set = set()
    while len(positions) < count:
        positions.update([(uniform(-1, 1), uniform(-1, 1))])
    return positions
