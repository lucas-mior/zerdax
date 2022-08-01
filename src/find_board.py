import cv2
import numpy as np
import math
import sys

from auxiliar import *
from lines import HoughBundler
import lffilter as lf
import random

def find_board(img):
    print("applying filter to image...")
    img.filt = lf.ffilter(img.sgray)

    print("applying distributed histogram equalization to image...")
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    img.clahe = clahe.apply(img.filt)

    print("finding region containing chess board...")
    img = find_region(img)

    img = bound_region(img)

    save(img, "hull", img.hull)

    img.canny = find_canny(img.clahe, wmin = 9)
    img.angles, img.select_lines = find_angles(img)

    lines,inter = magic_lines(img)
    exit()

    img.corners = find_corners(img, inter)

    return img

def bound_region(img):
    x,y,w,h = cv2.boundingRect(img.hullxy)
    limx = np.zeros((2), dtype='int32')
    limy = np.zeros((2), dtype='int32')
    limx[0] = max(y-60, 0)
    limx[1] = min(y+h+25, img.swidth)
    limy[0] = max(x-25, 0)
    limy[1] = max(x+w+25, img.sheigth)

    img.medges = img.medges[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.filt = img.filt[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.clahe = img.clahe[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    limx[0] = round(limx[0] / img.sfact)
    limx[1] = round(limx[1] / img.sfact)
    limy[0] = round(limy[0] / img.sfact)
    limy[1] = round(limy[1] / img.sfact)

    img.hull = img.gray[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hullcolor = img.color[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hxoff = limx[0]
    img.hyoff = limy[0]
    img = reduce_hull(img)
    img.hull3ch = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR)

    return img

def find_region(img, skip=False):
    got_hull = False
    img.filt = lf.ffilter(img.clahe)
    wc = 5
    Amin = round(0.5 * img.sarea)
    img.help = np.copy(img.filt) * 0
    while wc <= 11:
        print("Área mínima:", Amin)
        print("Canny wc:", wc)
        img.canny = find_canny(img.filt, wmin=wc)
        img, a = find_morph(img, Amin)

        drawn_contours = np.empty(img.gray3ch.shape, dtype='uint8') * 0
        cv2.drawContours(drawn_contours, img.cont,     -1, (255, 0, 0), thickness=3)
        cv2.drawContours(drawn_contours, [img.hullxy], -1, (0, 255, 0), thickness=3)
        # cv2.drawContours(drawn_contours, [img.poly],   -1, (0, 0, 255), thickness=3)
        img.help = cv2.bitwise_or(drawn_contours[:,:,0], drawn_contours[:,:,1])

        if a > Amin:
            print("{} > {} : {}".format(a, Amin, a/Amin))
            got_hull = True
            break
        else:
            print("problema é area")

        Amin = max(0.1*img.sarea, round(Amin - 0.03*img.sarea))
        wc += 1

    drawn_contours = cv2.addWeighted(img.gray3ch, 0.4, drawn_contours, 0.7, 0)
    save(img, "contours", drawn_contours)
    save(img, "medgesforcontour", img.medges)

    if not got_hull:
        print("finding board region failed")
        exit()

    return img

def find_morph(img, Amin):
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kd = 10
    kx = kd+round(kd/3)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (kd,kx))
    img.dilate = cv2.morphologyEx(img.filt, cv2.MORPH_DILATE, k_dil)
    img.divide = cv2.divide(img.filt, img.dilate, scale = 255)
    edges = cv2.threshold(img.divide, 0, 255, cv2.THRESH_OTSU)[1]
    edges = cv2.bitwise_not(edges)
    edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, ko, iterations = 1)
    edges = cv2.bitwise_or(edges, img.canny)
    edges = cv2.bitwise_or(edges, img.help)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    img.cont = contours[np.argmax(areas)]
    img.hullxy = cv2.convexHull(img.cont)
    arclen = cv2.arcLength(img.hullxy,True)
    # img.poly = cv2.approxPolyDP(img.hullxy,0.05*arclen,True)
    a = cv2.contourArea(img.hullxy)
    img.medges = edges

    # save(img, "dilate", img.dilate)
    # save(img, "divide", img.divide)

    return img, a

def find_canny(image, wmin = 6):
    c_thrl0 = 80
    c_thrh0 = 200
    c_thrl = c_thrl0
    c_thrh = c_thrh0

    while c_thrh > 30:
        canny = cv2.Canny(image, c_thrl, c_thrh)
        w = canny.mean()
        if w > wmin:
            print("{0:0=.2f} > {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
            break
        else:
            if wmin - w < wmin:
                print("{0:0=.2f} < {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
        c_thrl = max(10, c_thrl - 9)
        c_thrh = max(30, c_thrh - 9)

    if w < wmin:
        print("Canny failed: {0:0=.2f} < {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
        exit()

    return canny

def find_angles(img):
    got_hough = False
    h_maxg0 = 50
    h_minl0 = round((img.hwidth + img.hheigth)*0.2)
    h_thrv0 = round(h_minl0 / 2)
    h_angl0 = np.pi / 360

    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    minlines = 24
    while h_angl < (np.pi / 180):
        th = 180*(h_angl/np.pi)
        lines = cv2.HoughLinesP(img.canny, 1, h_angl,  h_thrv,  None, h_minl, h_maxg)
        if lines is not None and lines.shape[0] >= minlines:
            print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],th, h_thrv, h_minl, h_maxg))
            lines = radius_theta(lines)
            lines = filter_lines(img, lines)
            lines, angles = lines_kmeans(img, lines)
            print("angles: ", angles)
            got_hough = True
            break
        elif lines is not None:
            if th > random.uniform(0, th*4):
                print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],th, h_thrv, h_minl, h_maxg))
        if h_minl > h_minl0 / 2:
            h_minl -= 10
            h_thrv = round(h_minl / 2)
            
        h_angl += np.pi / 14400

    if not got_hough:
        print("find_angles failed")
        exit()

    drawn_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            cv2.line(drawn_lines,(x1,y1),(x2,y2), (0,0,250), 3)
    drawn_lines = cv2.addWeighted(img.hull3ch, 0.4, drawn_lines, 0.7, 0)
    save(img, "select", drawn_lines)

    return angles, lines

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

            dist = cv2.pointPolygonTest(img.shull, (x, y), True)
            if dist < -20:
                j += 1
                continue
            elif x > img.hwidth or y > img.hheigth or x < 0 or y < 0:
                j += 1
                continue
            else:
                j += 1
                if radius(last[0], last[1], x, y) > 10:
                    inter.append((x,y))
                    last = (x,y)
                else:
                    continue
        i += 1

    inter = np.array(inter, dtype='int32')
    return inter

def reduce_hull(img):
    img.hwidth = 900
    img.hfact = img.hwidth / img.hull.shape[1]
    img.hheigth = round(img.hfact * img.hull.shape[0])

    img.hull = cv2.resize(img.hull, (img.hwidth, img.hheigth))
    img.hullcolor = cv2.resize(img.hullcolor, (img.hwidth, img.hheigth))
    img.medges = cv2.resize(img.medges, (img.hwidth, img.hheigth))
    img.filt = cv2.resize(img.filt, (img.hwidth, img.hheigth))
    img.clahe = cv2.resize(img.clahe, (img.hwidth, img.hheigth))
    img.harea = img.hwidth * img.hheigth
    return img

def update_hull(img):
    contours, _ = cv2.findContours(img.medges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cont = contours[max_index]
    hullxy = cv2.convexHull(cont)

    drawn_contours = np.empty(img.hull3ch.shape, dtype='uint8') * 0
    cv2.drawContours(drawn_contours, [hullxy], -1, (0, 255, 0), thickness=3)
    cv2.drawContours(drawn_contours, cont, -1, (255, 0, 0), thickness=3)
    drawn_contours = cv2.addWeighted(img.hull3ch, 0.4, drawn_contours, 0.7, 0)
    save(img, "updatehull", drawn_contours)
    return hullxy

def magic_lines(img):
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_DILATE, k_dil)
    save(img, "cannylast", img.canny)
    got_hough = False
    h_maxg0 = 0
    h_minl0 = 70
    h_thrv0 = 60
    h_angl0 = np.pi / 180

    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    minlines = 40
    while h_angl < (np.pi / 60):
        th = 180*(h_angl/np.pi)
        lines = cv2.HoughLinesP(img.canny, 1, h_angl,  h_thrv,  None, h_minl, h_maxg)
        if lines is not None:
            lines = radius_theta(lines)
            lines = filter_lines(img, lines)
            lines = filter_angles(img, lines)
            if lines.shape[0] >= minlines:
                print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],th, h_thrv, h_minl, h_maxg))
                got_hough = True
                break
            else:
                if th > random.uniform(0, th*4):
                    print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],th, h_thrv, h_minl, h_maxg))
        if h_minl > h_minl0 / 1.5:
            h_minl -= 1
            h_thrv -= 1
            
        h_angl += np.pi / 7200

    if got_hough:
        drawn_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        draw_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0

        for line in lines:
            for x1,y1,x2,y2,r,t in line:
                cv2.line(draw_lines,(x1,y1),(x2,y2), (0,0,255), 3)
        drawn_lines = cv2.addWeighted(img.hull3ch, 0.4, draw_lines, 0.7, 0)
        save(img, "hough_magic", drawn_lines)

        dummy = np.copy(img.select_lines[:,:,0:6])
        lines = np.append(lines, dummy, axis=0)
        lines = filter_angles(img, lines)

        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        img.medges = cv2.morphologyEx(img.medges, cv2.MORPH_CLOSE, ko, iterations = 1)
        img.medges = cv2.bitwise_or(img.medges, draw_lines[:,:,0])
        img.shull = update_hull(img)
        inter = find_intersections(img, lines[:,0,:])

        drawn_circles = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        for p in inter:
            cv2.circle(drawn_circles, p, radius=7, color=(255, 0, 0), thickness=-1)
        drawn_circles = cv2.addWeighted(img.hull3ch, 0.4, drawn_circles, 0.7, 0)
        save(img, "intersections", drawn_circles)
        exit()
    else:
        print("FAILED @ {}, {}, {}, {}".format(180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        exit()
    return lines,inter

def filter_lines(img, lines):
    rem = np.empty(lines.shape[0])
    rem = np.int32(rem)

    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            if x1 < 5 and x2 < 5 or y1 < 5 and y2 < 5:
                rem[i] = 1
            elif (img.hwidth - x1) < 5 and (img.hwidth - x2) < 5 or (img.hheigth - y1) < 5 and (img.hheigth - y2) < 5:
                rem[i] = 1
            elif (x1 < 5 or (img.hwidth - x1) < 5) and (y2 < 5 or (img.hheigth - y2) < 5):
                rem[i] = 1
            elif (x2 < 5 or (img.hwidth - x2) < 5) and (y1 < 5 or (img.hheigth - y1) < 5):
                rem[i] = 1
            else:
                rem[i] = 0
        i += 1

    A = lines[rem==0]
    lines = A
    return lines

def filter_angles(img, lines, tol = 15):
    rem = np.empty(lines.shape[0])
    rem = np.int32(rem)

    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            if abs(t - img.angles[0]) > tol and abs(t - img.angles[1]) > tol:
                if img.angles.shape[0] == 2:
                    rem[i] = 1
                elif abs(t - img.angles[2]) > tol:
                    rem[i] = 1
                else:
                    rem[i] = 0
            else:
                rem[i] = 0
        i += 1

    A = lines[rem==0]
    lines = A
    return lines

def lines_kmeans(img, lines):
    lines = np.float32(lines)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(lines[:,:,5], 3, None, criteria, 10, flags)

    A = lines[labels==0]
    B = lines[labels==1]
    C = lines[labels==2]

    d1 = abs(centers[0] - centers[1])
    d2 = abs(centers[0] - centers[2])
    d3 = abs(centers[1] - centers[2])

    dd1 = d1 < 22.5 and d2 > 22.5 and d3 > 22.5
    dd2 = d2 < 22.5 and d1 > 22.5 and d3 > 22.5
    dd3 = d3 < 22.5 and d1 > 22.5 and d2 > 22.5

    if dd1 or dd2 or dd3:
        compactness,labels,centers = cv2.kmeans(lines[:,:,5], 2, None, criteria, 10, flags)
        A = lines[labels==0]
        B = lines[labels==1]

    diff = []
    diff.append((abs(centers[0] - 90), -90))
    diff.append((abs(centers[0] + 90), +90))
    diff.append((abs(centers[1] - 90), -90))
    diff.append((abs(centers[1] + 90), +90))
    if centers.shape[0] > 2:
        diff.append((abs(centers[2] - 90), -90))
        diff.append((abs(centers[2] + 90), +90))

    for d,k in diff:
        if d < 20:
            if abs(centers[0] - k) > 20 and abs(centers[1] - k):
                centers = np.append(centers, k)
            elif len(centers) > 2:
                if abs(centers[2] - k) > 20:
                    centers = np.append(centers, k)
            break

    lines = np.int32(lines)
    return lines, centers

def find_corners(img, inter):
    psum = np.empty((inter.shape[0], 3), dtype='int32')
    psub = np.empty((inter.shape[0], 3), dtype='int32')

    psum[:,0] = inter[:,0]
    psum[:,1] = inter[:,1]
    psum[:,2] = inter[:,0] + inter[:,1]
    psub[:,0] = inter[:,0]
    psub[:,1] = inter[:,1]
    psub[:,2] = inter[:,0] - inter[:,1]

    BR = psum[np.argmax(psum[:,2])]
    BL = psub[np.argmax(psub[:,2])]
    TR = psub[np.argmin(psub[:,2])]
    TL = psum[np.argmin(psum[:,2])]

    BR = BR[0:2]
    BL = BL[0:2]
    TR = TR[0:2]
    TL = TL[0:2]

    dummy = TR
    TR = BL
    BL = dummy

    drawn_circles = np.copy(img.hull3ch) * 0
    for p in BR, BL, TR, TL:
        cv2.circle(drawn_circles, p, radius=7, color=(0, 255, 0), thickness=-1)

    drawn_circles = cv2.addWeighted(img.hull3ch, 0.4, drawn_circles, 0.7, 0)
    save(img, "corners", drawn_circles)

    corners = np.array([BR, BL, TR, TL])
    print("board corners:", corners)

    return corners
