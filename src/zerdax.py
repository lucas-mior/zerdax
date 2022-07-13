#!/usr/bin/python

import argparse
import algo

def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')
    parser.add_argument('image',              type=str, default=None, help='Nome do arquivo da foto')
    parser.add_argument('-c_thrl', nargs='?', type=int, default=20,   help='Canny: low threshold')
    parser.add_argument('-c_thrh', nargs='?', type=int, default=120,  help='Canny: high threshold')
    parser.add_argument('-h_thrv', nargs='?', type=int, default=80,   help='Hough: minimum votes')
    parser.add_argument('-h_minl', nargs='?', type=int, default=250,  help='Hough: minimum line length')
    parser.add_argument('-h_maxg', nargs='?', type=int, default=30,   help='Hough: maximum gap')
    parser.add_argument('-savein', action=argparse.BooleanOptionalAction,default=False,help='Save intermediate steps')
    args = parser.parse_args()
    return args

def Main():
    args = parseargs()
    image = args.image
    print("======= Zerdax {} =======".format(image))
    print("savein = ", args.savein)
    fen = algo.full(image, args.c_thrl, args.c_thrh, args.h_thrv, args.h_minl, args.h_maxg, args.savein)
    print("FEN: ", fen)

if __name__ == '__main__':
    Main()
