#!/usr/bin/python

import argparse
import algo

def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')

    parser.add_argument('image',             type=str, default=None, help='Nome do arquivo da foto')
    parser.add_argument('-dummy', nargs='?', type=int, default=30,    help='Dummy argument')
    parser.add_argument('-savein', action=argparse.BooleanOptionalAction,default=False,help='Save intermediate steps')

    args = parser.parse_args()
    return args

def Main():
    args = parseargs()
    image = args.image
    print("\033[01;38;1m======= Zerdax {} =======\033[0;m".format(image))
    fen = algo.full(image, args.savein)
    print("FEN: ", fen)

if __name__ == '__main__':
    Main()
