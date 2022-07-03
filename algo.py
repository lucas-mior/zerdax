import cv2
import numpy as np

def testes(cvgray):
    scale_percent = 25
    width = int(cvgray.shape[1] * scale_percent / 100)
    height = int(cvgray.shape[0] * scale_percent / 100)
    dsize = (width, height)
    cvres = cv2.resize(cvgray, dsize)

    cvgauss = cv2.GaussianBlur(cvgray, [5,5], 0)
    cvmedian = cv2.medianBlur(cvgray, 5)

    kernel = np.array([[-1.0, -1.0, 2.0], 
                       [-1.0, 2.0, -1.0],
                       [2.0, -1.0, -1.0]])

    kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)

    cvhf = cv2.filter2D(cvres, -1, kernel)

    ret, cvthr = cv2.threshold(cvres, 160, 255, cv2.THRESH_BINARY)

    cveq = cv2.equalizeHist(cvres)

    for ts in range(1,16):
        for cl in range(1,5):
            clahe = cv2.createCLAHE(clipLimit=float(cl), tileGridSize=(ts,ts))
            cvclahe = clahe.apply(cvgray)
            cv2.imwrite('{}{}cvclahe.jpg'.format(ts, cl), cvclahe)

    # cv2.findChessboardCorners()

    # cv2.imwrite('cvgauss.jpg', cvgauss)
    cv2.imwrite('cvmedian.jpg', cvmedian)
    # cv2.imwrite('cveq.jpg', cveq)
    # cv2.imwrite('cvthr.jpg', cvthr)
    # cv2.imwrite('cvhf.jpg', cvhf)
    # cv2.imwrite('cvres.jpg', cvres) 
    # cv2.imwrite('cvgray.jpg', cvgray)

def full(image):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
    predicted_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    print("algo.full: {}".format(image))
    cvcolor = cv2.imread(image)
    cvgray = cv2.cvtColor(cvcolor, cv2.COLOR_BGR2GRAY)

    testes(cvgray)

    return predicted_fen
