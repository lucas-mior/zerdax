import cv2
import numpy as np

from Image import Image

from aux import *
from find_board import find_board
from find_squares import find_squares
from find_pieces import find_pieces

def create_fen(img):
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
        if i >= 1:
            fen += '/'

    img.fen = fen
    print("long fen:", img.fen)
    return img

def compress_fen(img):
    """ From: 11111q1k/1111r111/111p1pQP/111P1P11/11prn1R1/11111111/111111P1/R11111K1
        To: 5q1k/4r3/3p1pQP/3P1P2/2prn1R1/8/6P1/R5K1
    """
    print("generating compressed FEN...")
    fen = img.fen
    for length in reversed(range(2,9)):
        fen = fen.replace(length * '1', str(length))

    img.fen = fen
    return img

def reduce(img):
    img.swidth = 1000
    img.sfact = img.swidth / img.gray.shape[1]
    img.sheigth = round(img.sfact * img.gray.shape[0])

    img.sgray = cv2.resize(img.gray, (img.swidth, img.sheigth))
    img.sarea = img.sheigth * img.swidth
    return img

def full(filename):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
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
    img = create_fen(img)
    img = compress_fen(img)

    return img.fen
