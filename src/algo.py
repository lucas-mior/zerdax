import cv2
import numpy as np

from Image import Image
import glob

from find_board import find_board

def reduce():
    new_width = 1000
    img.fact = new_width / img.gray.shape[1]
    new_height = round(img.fact * img.gray.shape[0])

    dsize = (new_width, new_height)
    img.small = cv2.resize(img.gray, dsize)
    return

def full(filename):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
    img = Image(filename)

    reduce()
    img.board = find_board()

    print("board:", img.board)

    predicted_fen = "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    return predicted_fen
