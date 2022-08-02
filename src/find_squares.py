import cv2
import numpy as np
import math
import sys

from auxiliar import *
from lines import HoughBundler
import lffilter as lf
import random

def find_squares(img):
    print("generating 3 channel gray warped image for drawings...")
    img.warped3ch = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR)
    print("filtering warped image...")
    img.wfilt = lf.ffilter(img.warped)

    print("finding edges on warped image...")
    img.wcanny = find_wcanny(img, wmin = 12)

    print("finding vertical and horizontal lines...")
    vert,hori = w_lines(img)
    print("adjusting vertical and horizontal lines...")
    vert,hori = magic_vert_hori(img, vert, hori)

    save_lines(img, "verthori", vert, hori)

    print("calculating intersections...")
    inter = calc_intersections(img, vert[:,0,:], hori[:,0,:])
    if inter.shape[0] != 81:
        print("There should be exacly 81 intersections")
        exit()

    squares = calc_squares(img, inter)
    squares = np.float32(squares)
    sqback = np.zeros(squares.shape, dtype='float32')
    exit()

    for i in range(0, 8):
        sqback[i] = cv2.perspectiveTransform(squares[i], img.warpInvMatrix)

    img.sqback = np.int32(sqback)

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
            logprint(img, "{0:0=.2f} > {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
            break
        elif w > random.uniform(0, w*2):
            logprint(img, "{0:0=.2f} < {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
        if c_thrl > 10:
            c_thrl -= 9
        c_thrh -= 9

    return img.wcanny

def w_lines(img):
    got_hough = False
    h_minl0 = round((img.wwidth)*0.8)
    h_thrv0 = round(h_minl0 / 1.5)
    h_maxg0 = round(h_minl0 / 50) + 60
    h_angl0 = np.pi / 120

    tuned = 0
    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    j = 0
    minlines = 10
    while h_angl < (np.pi / 30):
        th = 180*(h_angl/np.pi)
        lines = cv2.HoughLinesP(img.wcanny, 2, h_angl, h_thrv,  None, h_minl, h_maxg)
        if lines is not None:
            lines = radius_theta(lines)
            lines = filter_90(img, lines)
            if lines.shape[0] > 10:
                bundler = HoughBundler()
                lines = bundler.process_lines(lines)
                lines = radius_theta(lines)
                vert,hori = geo_lines(img,lines)
                if vert.shape[0] >= minlines and hori.shape[0] >= minlines:
                    logprint(img, "{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],th, h_thrv, h_minl, h_maxg))
                    got_hough = True
                    break
            if th > random.uniform(0, th*4):
                logprint(img, "{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        j += 1
        h_angl += np.pi / 1800
        if h_angl > (np.pi / 40) and tuned < 20:
            h_angl = h_angl0
            h_minl -= 1
            h_thrv -= 8
            h_maxg += 5
            tuned += 1
            if tuned > 12:
                minlines = 9

    if not got_hough:
        print("FAILED @ {}, {}, {}, {}".format(180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        exit()

    return vert,hori

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

def get_distances(vert,hori):
    distv = np.zeros((vert.shape[0], 2), dtype='int32')
    x1 = (vert[1,0,0]+vert[1,0,2])/2
    x2 = (vert[0,0,0]+vert[0,0,2])/2
    distv[0,0] = abs(x1 - x2)
    distv[0,1] = abs(x1 - x2)
    for i in range (1, vert.shape[0]-1):
        x1 = (vert[i-1,0,0]+vert[i-1,0,2])/2
        x2 = (vert[i+0,0,0]+vert[i+0,0,2])/2
        x3 = (vert[i+1,0,0]+vert[i+1,0,2])/2
        distv[i,0] = abs(x1 - x2)
        distv[i,1] = abs(x1 - x2)
    i += 1
    x1 = (vert[i-1,0,0]+vert[i-1,0,2])/2
    x2 = (vert[i+0,0,0]+vert[i+0,0,2])/2
    distv[i,0] = abs(x1 - x2)
    distv[i,1] = abs(x1 - x2)

    disth = np.zeros((hori.shape[0], 2), dtype='int32')
    x1 = (hori[1,0,1]+hori[1,0,3])/2
    x2 = (hori[0,0,1]+hori[0,0,3])/2
    disth[0,0] = abs(x1 - x2)
    disth[0,1] = abs(x1 - x2)
    for i in range (1, hori.shape[0]-1):
        x1 = (hori[i-1,0,1]+hori[i-1,0,3])/2
        x2 = (hori[i+0,0,1]+hori[i+0,0,3])/2
        x3 = (hori[i+1,0,1]+hori[i+1,0,3])/2
        disth[i,0] = abs(x1 - x2)
        disth[i,1] = abs(x1 - x2)
    i += 1
    x1 = (hori[i-1,0,1]+hori[i-1,0,3])/2
    x2 = (hori[i+0,0,1]+hori[i+0,0,3])/2
    disth[i,0] = abs(x1 - x2)
    disth[i,1] = abs(x1 - x2)

    return distv, disth

def calc_intersections(img, vert, hori):
    inter = []
    last = (0,0)

    i = 0
    for x1,y1,x2,y2,r,t in vert:
        l1 = [(x1,y1), (x2,y2)]
        j = 0
        for xx1,yy1,xx2,yy2,rr,tt in hori:
            l2 =  [(xx1,yy1), (xx2,yy2)]
            if (x1,y1) == (xx1,yy1) and (x2,y2) == (xx2,yy2):
                continue

            xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
            ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

            div = determinant(xdiff, ydiff)
            if div == 0:
                j += 1
                continue

            d = (determinant(*l1), determinant(*l2))
            x = round(determinant(d, xdiff) / div)
            y = round(determinant(d, ydiff) / div)

            if x > img.wwidth or y > img.wheigth or x < 0 or y < 0:
                j += 1
                continue
            else:
                j += 1
                if radius(last[0], last[1], x, y) > 10:
                    inter.append((x,y))
                    last = (x,y)
        i += 1

    drawn_circles = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR) * 0
    for p in inter:
        cv2.circle(drawn_circles, p, radius=5, color=(255,0,0), thickness=-1)
    drawn_circles = cv2.addWeighted(img.warped3ch, 0.5, drawn_circles, 0.5, 1)
    save(img, "interboard", drawn_circles)

    inter = np.array(inter, dtype='int32')
    return inter

def mean_dist(distv,disth):
    medv1 = np.median(distv[1:-1,0])
    medv2 = np.median(distv[1:-1,1])
    medv = round((medv1 + medv2)/2)

    medh1 = np.median(disth[1:-1,0])
    medh2 = np.median(disth[1:-1,1])
    medh = round((medh1 + medh2)/2)

    return medv,medh

def wrong_lines(dist, med):
    rem = np.empty(dist.shape[0])
    rem = np.int32(rem)
    rem[:] = 0

    i = 0
    for d in dist:
        if abs(d[0] - med) > 8:
            if abs(d[1] - med) > 8:
                rem[i] = 1
            else:
                rem[i] = 0
        i += 1
    return rem

def right_lines(dist, med):
    cer = np.empty(dist.shape[0])
    cer = np.int32(cer)
    cer[:] = 0

    i = 0
    for d in dist:
        if abs(d[0] - med) < 8 and abs(d[1] - med) < 8:
            cer[i] = 1
        else:
            cer[i] = 0
        i += 1
    return cer

def magic_vert_hori(img, vert, hori):
    distv, disth = get_distances(vert,hori)
    medv, medh = mean_dist(distv,disth)

    remv = wrong_lines(distv, medv)
    remh = wrong_lines(disth, medh)

    vert = vert[remv==0]
    hori = hori[remh==0]

    distv, disth = get_distances(vert,hori)
    medv, medh = mean_dist(distv,disth)

    cerv = right_lines(distv, medv)
    cerh = right_lines(disth, medh)

    vert = vert[cerv==1]
    hort = hori[cerh==1]

    while vert[0,0,0] > (medv + 10):
        new = np.array([[[vert[0,0,0]-medv, 10, vert[0,0,0]-medv, 500, 0,0]]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[vert[:,0,0].argsort()]
    while abs(vert[-1,0,0] - 512) > (medv + 10):
        new = np.array([[[vert[-1,0,0]+medv, 10, vert[-1,0,0]+medv,500, 0,0]]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[vert[:,0,0].argsort()]
    while hori[0,0,1] > (medh + 10):
        new = np.array([[[10, hori[0,0,1]-medh, 500, hori[0,0,1]-medh, 0,0]]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[hori[:,0,1].argsort()]
    while abs(hori[-1,0,1] - 512) > (medh + 10):
        new = np.array([[[10, hori[-1,0,1]+medh, 500, hori[-1,0,1]+medh, 0,0]]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[hori[:,0,1].argsort()]

    i = 0
    while i < (vert.shape[0] - 1):
        if abs(vert[i,0,0] - vert[i+1,0,0]) > (medv*1.5):
            new = np.array([[[vert[i,0,0]+medv, 10, vert[i,0,0]+medv, 500, 0,0]]], dtype='int32')
            vert = np.append(vert, new, axis=0)
            vert = vert[vert[:,0,0].argsort()]
        i += 1

    i = 0
    while i < (hori.shape[0] - 1):
        if abs(hori[i,0,1] - hori[i+1,0,1]) > (medh*1.5):
            new = np.array([[[10, hori[i,0,1]+medh, 500, hori[i,0,1]+medh, 0,0]]], dtype='int32')
            hori = np.append(hori, new, axis=0)
            hori = hori[hori[:,0,1].argsort()]
        i += 1

    if vert.shape[0] == 10:
        d1 = abs(vert[0,0,0]-0)
        d2 = abs(vert[-1,0,0]-512)
        if d1 < d2:
            vert = vert[1:]
        else:
            vert = vert[0:-1]
    elif vert.shape[0] == 11:
        vert = vert[1:-1]

    if hori.shape[0] == 10:
        d1 = abs(hori[0,0,1]-0)
        d2 = abs(hori[-1,0,1]-512)
        if d1 < d2:
            hori = hori[1:]
        else:
            hori = hori[0:-1]
    elif hori.shape[0] == 11:
        hori = hori[1:-1]

    return vert, hori

def calc_squares(img, inter):
    inter = inter[inter[:,0].argsort()]
    intersq = np.zeros((9,9,2), dtype='int32')
    interA = inter[00:9]  # A
    interB = inter[9:18]  # B
    interC = inter[18:27] # C
    interD = inter[27:36] # D
    interE = inter[36:45] # E
    interF = inter[45:54] # F
    interG = inter[54:63] # G
    interH = inter[63:72] # H
    interZ = inter[72:81] # right

    intersq[0,:] = interA[interA[:,1].argsort()[::-1]] # A
    intersq[1,:] = interB[interB[:,1].argsort()[::-1]] # B
    intersq[2,:] = interC[interC[:,1].argsort()[::-1]] # C
    intersq[3,:] = interD[interD[:,1].argsort()[::-1]] # D
    intersq[4,:] = interE[interE[:,1].argsort()[::-1]] # E
    intersq[5,:] = interF[interF[:,1].argsort()[::-1]] # F
    intersq[6,:] = interG[interG[:,1].argsort()[::-1]] # G
    intersq[7,:] = interH[interH[:,1].argsort()[::-1]] # H
    intersq[8,:] = interZ[interZ[:,1].argsort()[::-1]] # right

    squares = np.zeros((8,8,4,2), dtype='int32')
    for i in range(0,8):
        for j in range(0,8):
            squares[i,j,0] = intersq[i,j]
            squares[i,j,1] = intersq[i+1,j]
            squares[i,j,2] = intersq[i+1,j+1]
            squares[i,j,3] = intersq[i,j+1]

    drawn_lines = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR) * 0
    cv2.drawContours(drawn_lines, [squares[0,0]], -1, color=(255,0,0), thickness=2) #A1
    cv2.drawContours(drawn_lines, [squares[4,3]], -1, color=(0,255,0), thickness=2) #E4
    cv2.drawContours(drawn_lines, [squares[2,4]], -1, color=(0,0,255), thickness=2) #C5
    drawn_contours = cv2.addWeighted(img.warped3ch, 0.5, drawn_lines, 0.5, 1)
    save(img, "A1E4C5", drawn_contours)

    return squares

def save_lines(img, name, vert, hori):
    drawn_lines = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR) * 0
    draw_lines = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR) * 0
    for line in vert:
        for x1,y1,x2,y2,r,t in line:
            cv2.line(draw_lines,(x1,y1),(x2,y2), color=(255,0,0), thickness=3)
    for line in hori:
        for x1,y1,x2,y2,r,t in line:
            cv2.line(draw_lines,(x1,y1),(x2,y2), color=(0,255,0), thickness=3)
    drawn_lines = cv2.addWeighted(img.warped3ch, 0.5, draw_lines, 0.5, 1)
    save(img, name, drawn_lines)
