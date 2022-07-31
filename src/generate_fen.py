import cv2

def generate_fen(img):
    img = create_fen(img)
    img.fen = compress_fen(img.fen)
    return img

def create_fen(img):
    fen = ''
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = img.sqback[j,i]
            got_piece = False
            print("sq: ", i, j)
            for piece in img.pieces:
                p = (int(piece[4]), int(piece[2]) - 15)
                if cv2.pointPolygonTest(sq, p, True) >= 0:
                    fen += piece[6].split(" ")[0]
                    got_piece = True
                    img.pieces.remove(piece)
                    break
            if not got_piece:
                fen += '1'
        if i >= 1:
            fen += '/'

    img.fen = fen
    print("long fen:", img.fen)
    return img

def compress_fen(fen):
    print("generating compressed FEN...")
    for length in reversed(range(2,9)):
        fen = fen.replace(length * '1', str(length))

    return fen
