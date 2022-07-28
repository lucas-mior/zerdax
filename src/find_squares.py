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
import random

def draww_lines(img, name, vert, hori):
    drawn_lines = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR) * 0
    draw_lines = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR) * 0
    for line in vert:
        for x1,y1,x2,y2,r,t in line:
            cv2.line(draw_lines,(x1,y1),(x2,y2),(255,0,0),round(2/img.sfact))
    for line in hori:
        for x1,y1,x2,y2,r,t in line:
            cv2.line(draw_lines,(x1,y1),(x2,y2),(0,255,0),round(2/img.sfact))
    drawn_lines = cv2.addWeighted(img.warped3ch, 0.5, draw_lines, 0.7, 0)
    save(img, name, drawn_lines)

def find_squares(img):
    img = perspective_transform(img)
    img.warped3ch = cv2.cvtColor(img.warped, cv2.COLOR_GRAY2BGR)

    img.wwang = lwang.wang_filter(img.warped)
    # save(img, "wwang", img.wwang)

    img.wcanny = find_wcanny(img, wmin = 12)
    save(img, "wcanny", img.wcanny)

    vert,hori = w_lines(img)
    draww_lines(img, "vert_hori0", vert, hori)

    distv, disth = get_distances(vert,hori)
    medv, medh = mean_dist(distv,disth)

    remv = iterate(distv, medv)
    remh = iterate(disth, medh)

    vert = vert[remv==0]
    hori = hori[remh==0]

    distv, disth = get_distances(vert,hori)
    medv, medh = mean_dist(distv,disth)

    cerv = iterate2(distv, medv)
    cerh = iterate2(disth, medh)

    vert = vert[cerv==1]
    hort = hori[cerh==1]
    draww_lines(img, "vert_hori1", vert, hori)

    while vert[0,0,0] > (medv + 10):
        new = np.array([[[vert[0,0,0]-medv, 10, vert[0,0,0]-medv, 400, 0,0]]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[vert[:,0,0].argsort()]
    while abs(vert[-1,0,0] - 412) > (medv + 10):
        new = np.array([[[vert[-1,0,0]+medv, 10, vert[-1,0,0]+medv,400, 0,0]]], dtype='int32')
        vert = np.append(vert, new, axis=0)
        vert = vert[vert[:,0,0].argsort()]
    while hori[0,0,1] > (medh + 10):
        new = np.array([[[10, hori[0,0,1]-medh, 400, hori[0,0,1]-medh, 0,0]]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[hori[:,0,1].argsort()]
    while abs(hori[-1,0,1] - 412) > (medh + 10):
        new = np.array([[[10, hori[-1,0,1]+medh, 400, hori[-1,0,1]+medh, 0,0]]], dtype='int32')
        hori = np.append(hori, new, axis=0)
        hori = hori[hori[:,0,1].argsort()]

    draww_lines(img, "vert_hori2", vert, hori)

    i = 0
    while i < (vert.shape[0] - 1):
        if abs(vert[i,0,0] - vert[i+1,0,0]) > (medv*1.5):
            new = np.array([[[vert[i,0,0]+medv, 10, vert[i,0,0]+medv, 400, 0,0]]], dtype='int32')
            vert = np.append(vert, new, axis=0)
            vert = vert[vert[:,0,0].argsort()]
        i += 1

    i = 0
    while i < (hori.shape[0] - 1):
        if abs(hori[i,0,1] - hori[i+1,0,1]) > (medh*1.5):
            new = np.array([[[10, hori[i,0,1]+medh, 400, hori[i,0,1]+medh, 0,0]]], dtype='int32')
            hori = np.append(hori, new, axis=0)
            hori = hori[hori[:,0,1].argsort()]
        i += 1

    draww_lines(img, "vert_hori3", vert, hori)

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
    while h_angl < (np.pi / 10):
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
                    print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],th, h_thrv, h_minl, h_maxg))
                    got_hough = True
                    break

        if lines is not None:
            if th > random.uniform(0, th*4):
                print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        j += 1
        h_angl += np.pi / 1800
        if h_angl > (np.pi / 20) and tuned < 20:
            h_angl = h_angl0
            h_minl -= 1
            h_thrv -= 8
            h_maxg += 5
            tuned += 1
            if tuned > 12:
                minlines = 9

    if not got_hough:
        print("FAILED @ {}, {}, {}, {}".format(180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        save(img, "wcannyfail", img.wcanny)
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

def geo_lines(img, lines):
    lines = radius_theta(lines)
    vert = lines[abs(lines[:,0,5]) > 45]
    hori = lines[abs(lines[:,0,5]) < 45]

    vert = vert[vert[:,0,0].argsort()]
    hori = hori[hori[:,0,1].argsort()]

    return vert, hori

def get_distances(vert,hori):
    distv = np.zeros((vert.shape[0], 2), dtype='int32')
    distv[0,0] = abs(vert[1,0,0] - vert[0,0,0])
    distv[0,1] = abs(vert[1,0,0] - vert[0,0,0])
    for i in range (1, vert.shape[0]-1):
        distv[i,0] = abs(vert[i-1,0,0] - vert[i,0,0])
        distv[i,1] = abs(vert[i+1,0,0] - vert[i,0,0])
        print("distv0: {} = abs({} - {})".format(distv[i,0], vert[i-1,0,0], vert[i,0,0]))
        print("distv1: {} = abs({} - {})".format(distv[i,1], vert[i+1,0,0], vert[i,0,0]))
    i += 1
    distv[i,0] = abs(vert[i-1,0,0] - vert[i,0,0])
    distv[i,1] = abs(vert[i-1,0,0] - vert[i,0,0])

    disth = np.zeros((hori.shape[0], 2), dtype='int32')
    disth[0,0] = abs(hori[1,0,1] - hori[0,0,1])
    disth[0,1] = abs(hori[1,0,1] - hori[0,0,1])
    for i in range (1, hori.shape[0]-1):
        disth[i,0] = abs(hori[i-1,0,1] - hori[i,0,1])
        disth[i,1] = abs(hori[i+1,0,1] - hori[i,0,1])
        print("disth0: {} = abs({} - {})".format(disth[i,0], hori[i-1,0,1], hori[i,0,1]))
        print("disth1: {} = abs({} - {})".format(disth[i,1], hori[i+1,0,1], hori[i,0,1]))
    i += 1
    disth[i,0] = abs(hori[i-1,0,1] - hori[i,0,1])
    disth[i,1] = abs(hori[i-1,0,1] - hori[i,0,1])

    print("distv:", distv)
    print("disth:", disth)

    return distv, disth

def find_intersections(img, lines):
    inter = []
    last = (0,0)

    i = 0
    for x1,y1,x2,y2,r,t in lines:
        l1 = [(x1,y1), (x2,y2)]
        j = 0
        for xx1,yy1,xx2,yy2,rr,tt in lines:
            l2 =  [(xx1,yy1), (xx2,yy2)]
            if (x1,y1) == (xx1,yy1) and (x2,y2) == (xx2,yy2):
                continue

            if abs(t - tt) < 20 or abs(t - tt) > 160:
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
                else:
                    # print("Close point ignored: ({},{}) ~ ({},{})".format(last[0],last[1],x,y))
                    continue
        i += 1

    inter = np.array(inter, dtype='int32')
    return inter

def mean_dist(distv,disth):
    medv1 = np.median(distv[1:-1,0])
    medv2 = np.median(distv[1:-1,1])
    medv = round((medv1 + medv2)/2)

    medh1 = np.median(disth[1:-1,0])
    medh2 = np.median(disth[1:-1,1])
    medh = round((medh1 + medh2)/2)

    print("medv: ", medv1, medv2, medv)
    print("medh: ", medh1, medh2, medh)
    return medv,medh

def iterate(dist, med):
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

def iterate2(dist, med):
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
