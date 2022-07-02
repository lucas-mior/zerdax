import cv2

def full(image):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
    predicted_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    print("algo.full: {}".format(image))
    cvcolor = cv2.imread(image)
    cvgray = cv2.cvtColor(cvcolor, cv2.COLOR_BGR2GRAY)

    scale_percent = 25
    width = int(cvgray.shape[1] * scale_percent / 100)
    height = int(cvgray.shape[0] * scale_percent / 100)
    dsize = (width, height)
    cvres = cv2.resize(cvgray, dsize)

    cv2.imwrite('resize.jpg', cvres) 
    cv2.imwrite('gray.jpg', cvgray)

    return predicted_fen
