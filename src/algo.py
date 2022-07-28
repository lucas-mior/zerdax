import cv2
import numpy as np

from Image import Image

from aux import *
from find_board import find_board
from find_squares import find_squares
from find_pieces import find_pieces
from produce_fen import produce_fen

def compressed_fen(fen):
    """ From: 11111q1k/1111r111/111p1pQP/111P1P11/11prn1R1/11111111/111111P1/R11111K1
        To: 5q1k/4r3/3p1pQP/3P1P2/2prn1R1/8/6P1/R5K1
    """
    for length in reversed(range(2,9)):
        fen = fen.replace(length * '1', str(length))
    return fen

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

    piece = img.ObjectsList[0]
    print("piece: ", piece)

    fen = ''
    for i in range(7, -1, -1): #FEN começa em cima
        for j in range(0, 8):
            sq = img.sqback[j,i]
            got_piece = False
            print("sq: ", i, j)
            for piece in img.ObjectsList:
                p = (int(piece[4]), int(piece[2]) - 15) # no Xmed, um pouco acima do Ymin
                if cv2.pointPolygonTest(sq, p, True) >= 0:
                    fen += piece[6].split(" ")[0]
                    got_piece = True
                    img.ObjectsList.remove(piece)
                    break
            if not got_piece:
                fen += '1'
        fen += '/'

    fen = compressed_fen(fen)
    # [top, left, bottom, right, mid_v, mid_h, label, scores]

    return fen
