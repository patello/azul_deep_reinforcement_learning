from game.azul import Azul

import math

class NNRunner:
    def __init__(self):
        self.game = Azul()
        self.move_counter = 0

def nn_serialize(display,color,pattern):
    return display+color*6+pattern*5*6

def nn_deserialize(i):
    display = i%6
    color = math.floor(i/6)%5
    pattern = math.floor(i/(5*6))
    return (display, color, pattern)