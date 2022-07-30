#!/usr/bin/python

import argparse
import algo

def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')

    parser.add_argument('image', type=str, default=None, help='Image filename')

    args = parser.parse_args()
    return args

def Main():
    args = parseargs()
    image = args.image
    print("\033[31;1;1m==========\033[37;1;1m zerdax {} \033[31;1;1m==========\033[0;m".format(image))
    fen = algo.full(image)
    print("\033[01;38;1mFEN:\033[0;m", fen)

if __name__ == '__main__':
    Main()
