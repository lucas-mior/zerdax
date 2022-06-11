import cv2

def full(image):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
    predicted_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    print("algo.full: {}".format(image))
    cvcolor = cv2.imread(image)
    cvgray = cv2.cvtColor(cvcolor, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("{}.jpg".format(image), cvgray)
    return predicted_fen
