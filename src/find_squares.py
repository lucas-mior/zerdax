import cv2
import numpy as np
import math
import sys
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from Image import Image
from aux import *
from lines import HoughBundler
import lwang

def find_squares(img):
    img = perspective_transform(img)
    img.warped3ch = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR)

    img.wwang = lwang.wang_filter(img.warped)
    # save(img, "wwang", img.wwang)

    img.wcanny = find_wcanny(img, wmin = 12)
    save(img, "wcanny", img.wcanny)

    lines = w_lines(img)
    lines = geo_lines(img, lines)
    return img

def perspective_transform(img):
    BR = img.corners[0] 
    BL = img.corners[1] 
    TR = img.corners[2] 
    TL = img.corners[3] 
    rect = np.array(((TL[0], TL[1]), (TR[0], TR[1]), (BR[0], BR[1]), (BL[0], BL[1])), dtype="float32")

    width = 412
    height = 412
    img.wwidth = width
    img.wheigth = width

    dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(rect,dst)
    img.warped = cv2.warpPerspective(img.hull, M, (width, height))

    # save(img, "warped", img.warped)
    return img

def find_wcanny(img, wmin = 12):
    c_thrl0 = 100
    c_thrh0 = 220
    c_thrl = c_thrl0
    c_thrh = c_thrh0

    while c_thrh > 20:
        img.wcanny = cv2.Canny(img.warped, c_thrl, c_thrh)
        w = img.wcanny.mean()
        if w > wmin:
            print("{0:0=.2f} > {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
            break
        else:
            if wmin - w < wmin:
                print("{0:0=.2f} < {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
        if c_thrl > 10:
            c_thrl -= 9
        c_thrh -= 9

    return img.wcanny

def w_lines(img):
    got_hough = False
    h_minl0 = round((img.wwidth)*0.7)
    h_thrv0 = round(h_minl0 / 2)
    h_maxg0 = round(h_minl0 / 50) + 50
    h_angl0 = np.pi / 120

    tuned = 0
    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    j = 0
    while h_angl < (np.pi / 10):
        lines = cv2.HoughLinesP(img.wcanny, 1, h_angl, h_thrv,  None, h_minl, h_maxg)
        if lines is not None:
            lines = radius_theta(lines)
            lines = filter_90(img, lines)
            bundler = HoughBundler() 
            lines = bundler.process_lines(lines) 
            lines = radius_theta(lines)
            if lines.shape[0] >= 18:
                print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
                got_hough = True
                break

        if lines is not None:
            print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        j += 1
        h_angl += np.pi / 1800
        if h_angl > (np.pi / 20) and tuned < 10:
            h_angl = h_angl0
            h_minl -= 5
            h_thrv -= 10
            h_maxg += 5
            tuned += 1

    if got_hough:
        drawn_lines = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR) * 0
        draw_lines = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR) * 0

        for line in lines:
            for x1,y1,x2,y2,r,t in line:
                cv2.line(draw_lines,(x1,y1),(x2,y2),(0,0,255),round(2/img.sfact))
        drawn_lines = cv2.addWeighted(img.warped3ch, 0.5, draw_lines, 0.8, 0)
        save(img, "hough_warped", drawn_lines)

    else:
        print("FAILED @ {}, {}, {}, {}".format(180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        exit()

    return lines

def filter_90(img, lines):
    rem = np.empty(lines.shape[0])
    rem = np.int32(rem)

    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            if abs(t - 90) > 5 and abs(t + 90) > 5 and abs(t) > 5:
                rem[i] = 1
            else:
                rem[i] = 0
        i += 1

    A = lines[rem==0]
    lines = A
    return lines

def geo_lines(img, lines):
    print("dummy geo_lines")
    return lines
