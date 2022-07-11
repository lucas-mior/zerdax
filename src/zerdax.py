#!/usr/bin/python

import argparse
import algo

def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')
    parser.add_argument('image',              type=str, default=None, help='Nome do arquivo da foto')
    parser.add_argument('-c_thrl', nargs='?', type=int, default=20,   help='Canny: low threshold')
    parser.add_argument('-c_thrh', nargs='?', type=int, default=120,  help='Canny: high threshold')
    parser.add_argument('-h_thrv', nargs='?', type=int, default=60,   help='Hough: minimum votes')
    parser.add_argument('-h_minl', nargs='?', type=int, default=150,  help='Hough: minimum line length')
    parser.add_argument('-h_maxg', nargs='?', type=int, default=15,   help='Hough: maximum gap')
    args = parser.parse_args()
    return args

def Main():
    args = parseargs()
    image = args.image
    print("======= Zerdax {} =======".format(image))
    fen = algo.full(image, args.c_thrl, args.c_thrh, args.h_thrv, args.h_minl, args.h_maxg)
    print(fen)

if __name__ == '__main__':
    Main()
