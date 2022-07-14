import cv2
import numpy as np

from Image import Image

from find_board import find_board

def reduce(img):
    img.swidth = 1000
    img.sfact = img.swidth / img.gray.shape[1]
    img.sheigth = round(img.sfact * img.gray.shape[0])

    img.small = cv2.resize(img.gray, (img.swidth, img.sheigth))
    img.sarea = img.sheigth * img.swidth
    return img

def full(filename, c_thl, c_thh, h_th, h_minl, h_maxg, save):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
    img = Image(filename)
    img.color = cv2.imread(filename)
    img.gray = cv2.cvtColor(img.color, cv2.COLOR_BGR2GRAY)
    img = reduce(img)
    img.gray3ch = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR)

    img.save = save
    img.board = find_board(img, c_thl, c_thh, h_th, h_minl, h_maxg)

    print("board:", img.board)

    predicted_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    return predicted_fen
