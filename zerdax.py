#!/usr/bin/python

import argparse
import algo

def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')
    parser.add_argument('image', type=str, help='Nome do arquivo da foto')
    args = parser.parse_args()
    return args

def Main():
    args = parseargs()
    image = args.image
    print("Processing: {}".format(image))
    fen = algo.full(image)
    print(fen)

if __name__ == '__main__':
    Main()
