import cv2
import numpy as np

from Image import Image

from find_board import find_board

def reduce(img):
    new_width = 1000
    img.fact = new_width / img.gray.shape[1]
    new_height = round(img.fact * img.gray.shape[0])

    dsize = (new_width, new_height)
    img.small = cv2.resize(img.gray, dsize)
    return img

def full(filename, c_thl, c_thh, h_th, h_minl, h_maxg):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
    img = Image(filename)
    img.color = cv2.imread(filename)
    img.gray = cv2.cvtColor(img.color, cv2.COLOR_BGR2GRAY)

    img = reduce(img)
    img.board = find_board(img, c_thl, c_thh, h_th, h_minl, h_maxg)

    print("board:", img.board)

    predicted_fen = "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    return predicted_fen
