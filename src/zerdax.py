#!/usr/bin/python

import argparse
from algo import full

def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')

    parser.add_argument('image', type=str, default=None, help='Image filename')

    args = parser.parse_args()
    return args

def Main():
    args = parseargs()
    image = args.image
    print("============ zerdax {} ============".format(image))
    fen = full(image)
    print("FEN:", fen)

if __name__ == '__main__':
    Main()
