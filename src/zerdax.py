#!/usr/bin/python

import argparse
import algo

def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')
    parser.add_argument('image', type=str, help='Nome do arquivo da foto')
    parser.add_argument('-a', nargs='?', type=int, default=20,  help='Canny: low threshold')
    parser.add_argument('-b', nargs='?', type=int, default=120, help='Canny: high threshold')
    parser.add_argument('-c', nargs='?', type=int, default=60,  help='Hough: votes')
    parser.add_argument('-d', nargs='?', type=int, default=150, help='Hough: minLen')
    parser.add_argument('-e', nargs='?', type=int, default=15,  help='Hough: maxGap')
    args = parser.parse_args()
    return args

def Main():
    args = parseargs()
    image = args.image
    print("Processing: {}".format(image))
    fen = algo.full(image, args.a, args.b, args.c, args.d, args.e)
    print(fen)

if __name__ == '__main__':
    Main()
