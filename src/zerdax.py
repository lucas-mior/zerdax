#!/usr/bin/python

import argparse
from algorithm import algorithm

def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')

    parser.add_argument('image', type=str, default=None, help='Image filename')
    parser.add_argument('-log', default=False, action='store_true', help='Detailed log')

    args = parser.parse_args()
    return args

def Main():
    args = parseargs()
    image = args.image
    log = args.log
    print("============ zerdax {} ============".format(image))
    fen = algorithm(image, log)
    print("FEN:", fen)

if __name__ == '__main__':
    Main()
