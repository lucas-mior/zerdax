import cv2
import numpy as np

def find_board(cvgray)
    fil = cv2.GaussianBlur(cvgray, (3,3), 0)
    can = cv2.Canny(fil, 100, 180)

    rho = 1             # distance resolution in pixels of the Hough grid
    theta = np.pi / 180 # angular resolution in radians of the Hough grid
    threshold = 15      # minimum number of votes (intersections in Hough grid cell)

    lines = cv2.HoughLinesP(can, rho, theta, threshold)
    print(image, lines[0][0])

    line_image = cv2.cvtColor(cvgray, cv2.COLOR_GRAY2BGR) * 0

    cvgray3ch = cv2.cvtColor(cvgray, cv2.COLOR_GRAY2BGR)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,250),10)

    hou = cv2.addWeighted(cvgray3ch, 1, line_image, 0.8, 0)

    cv2.imwrite("{}hough.jpg".format(image), hou)

def testes(name, cvgray):
    scale_percent = 25
    width = int(cvgray.shape[1] * scale_percent / 100)
    height = int(cvgray.shape[0] * scale_percent / 100)
    dsize = (width, height)
    cvres = cv2.resize(cvgray, dsize)

    kernel = np.array([[-1.0, -1.0, 2.0],
                       [-1.0, 2.0, -1.0],
                       [2.0, -1.0, -1.0]])

    kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)

    cvhf = cv2.filter2D(cvgray, -1, kernel)

    ret, cvthr = cv2.threshold(cvgray, 160, 255, cv2.THRESH_BINARY)

    cveq = cv2.equalizeHist(cvgray)

    ts = 5
    cl = 2

    cvgauss = cv2.GaussianBlur(cvgray, [5,5], 0)
    cvmedian = cv2.medianBlur(cvgray, 5)

    clahe = cv2.createCLAHE(clipLimit=float(cl), tileGridSize=(ts,ts))

    cvclahe_before = clahe.apply(cvgray)
    cvmedian = cv2.medianBlur(cvclahe_before, 5)
    cvclahe_after = clahe.apply(cvmedian)

    cv2.imwrite('{}clahe.jpg'.format(name), cvclahe_after)

def full(image):
    """ given a file path to a chessboard image,
        returns a FEN notation
    """
    print("algo.full: {}".format(image))
    cvcolor = cv2.imread(image)
    cvgray = cv2.cvtColor(cvcolor, cv2.COLOR_BGR2GRAY)

    testes(image, cvgray)

    predicted_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    return predicted_fen
