import cv2
import numpy as np
from pathlib import Path

from aux import *
from find_board import find_board
from find_squares import find_squares
from find_pieces import find_pieces
from fen import generate_fen

class Image:
    def __init__(self, filename):
        self.filename = filename
        self.basename = Path(self.filename).stem

def reduce(img):
    img.swidth = 1000
    img.sfact = img.swidth / img.gray.shape[1]
    img.sheigth = round(img.sfact * img.gray.shape[0])

    img.sgray = cv2.resize(img.gray, (img.swidth, img.sheigth))
    img.sarea = img.sheigth * img.swidth
    return img

def full(filename):
    img = Image(filename)
    print("reading image...")
    img.color = cv2.imread(img.filename)
    print("converting image to grayscale...")
    img.gray = cv2.cvtColor(img.color, cv2.COLOR_BGR2GRAY)
    print("reducing image to 1000 width...")
    img = reduce(img)
    print("generating 3 channel gray image for drawings...")
    img.gray3ch = cv2.cvtColor(img.sgray, cv2.COLOR_GRAY2BGR)

    img = find_board(img)
    img = find_squares(img)
    img = find_pieces(img)
    img = generate_fen(img)

    return img.fen
