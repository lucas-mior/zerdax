import cv2
import numpy as np

from Image import Image
from find_board import find_board

def reduce(img):
    new_width = 1000
    img.fact = new_width / img.gray.shape[1]
    new_height = round(img.fact * img.gray.shape[0])

    dsize = (new_width, new_height)
    return cv2.resize(img.gray, dsize)

def full(filename):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
    img = Image(filename)
    print(img.basename)

    img.small = reduce(img)
    # cv2.imwrite('{}0small.jpg'.format(img.filename), img.small)

    img.board = find_board(img)

    print(img.board)

    predicted_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    return predicted_fen
