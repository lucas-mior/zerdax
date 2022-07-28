import cv2
import numpy as np

from Image import Image

from aux import *
from find_board import find_board
from find_squares import find_squares
from find_pieces import find_pieces
from produce_fen import produce_fen

def reduce(img):
    img.swidth = 1000
    img.sfact = img.swidth / img.gray.shape[1]
    img.sheigth = round(img.sfact * img.gray.shape[0])

    img.sgray = cv2.resize(img.gray, (img.swidth, img.sheigth))
    img.sarea = img.sheigth * img.swidth
    return img

def full(filename, save):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
    img = Image(filename)
    img.color = cv2.imread(filename)
    img.gray = cv2.cvtColor(img.color, cv2.COLOR_BGR2GRAY)
    img = reduce(img)
    img.gray3ch = cv2.cvtColor(img.sgray, cv2.COLOR_GRAY2BGR)

    img.save = save
    img = find_board(img)
    img = find_squares(img)
    img = find_pieces(img)


    drawn_circles = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0

    fen = '' 
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = img.sqback[i,j]
            got_piece = False
            print("sq: ", sq)
            for piece in img.ObjectsList:
                p = (int(piece[3]), int(piece[5]))
                print("p: ", p)
                if cv2.pointPolygonTest(sq, p, True) >= 0:
                    fen += piece[6].split(" ")[0]
                    got_piece = True
                    img.ObjectsList.remove(piece)
            if got_piece:
                continue
            else:
                fen += '1'
        fen += '/'

    # [top, left, bottom, right, mid_v, mid_h, label, scores]

    return fen
