import cv2
import numpy as np
import math
import sys

from auxiliar import *
from lines import HoughBundler
import lffilter as lf
import random

def find_board(img):
    print("pre processing image...")
    img = pre_process(img)

    print("finding region containing chess board...")
    img = find_region(img)
    print("cropping image to fit only board region...")
    img = bound_region(img)

    # save(img, "fedges", img.fedges)
    # save(img, "hull", img.hull)
    save(img, "hullBGR", img.hullBGR)
    exit()

    img = find_angles(img)

    lines,inter = magic_lines(img)
    img.corners = find_corners(img, inter)
    exit()

    return img

def create_cannys(img, w = 6):
    print("finding edges for gray, S, V images...")
    img.cannyG = find_canny(img.claheG, wmin = w-1)
    # img.cannyS = find_canny(img.claheS, wmin = w-4)
    img.cannyV = find_canny(img.claheV, wmin = w+1)
    # img.canny = cv2.bitwise_or(img.cannyS, img.cannyV)
    img.canny = cv2.bitwise_or(img.cannyG, img.cannyV)
    return img

def bound_region(img):
    x,y,w,h = cv2.boundingRect(img.hullxy)
    limx = np.zeros((2), dtype='int32')
    limy = np.zeros((2), dtype='int32')
    limx[0] = max(y-20, 0)
    limx[1] = min(y+h+20, img.width)
    limy[0] = max(x-20, 0)
    limy[1] = max(x+w+20, img.heigth)

    img.medges = img.medges[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.G = img.G[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.claheG = img.claheG[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    # img.claheS = img.claheS[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.claheV = img.claheV[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.fedges = img.fedges[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hull = img.gray[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hullBGR = img.BGR[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hxoff = limx[0]
    img.hyoff = limy[0]
    img = reduce_hull(img)
    img.hull3ch = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR)

    return img

def find_region(img):
    got_hull = False
    Wc = 5
    W0 = 10
    Amin = round(0.5 * img.area)
    A0 = round(0.1 * img.area)
    img.help = np.copy(img.G) * 0
    while Wc <= W0 or Amin >= A0:
        print("Área mínima:", Amin)
        print("Canny Wc:", Wc)
        img = create_cannys(img, w = Wc)
        img, a = find_morph(img)

        drawn_contours = np.empty(img.gray3ch.shape, dtype='uint8') * 0
        cv2.drawContours(drawn_contours, img.cont,     -1, (255, 0, 0), thickness=1)
        cv2.drawContours(drawn_contours, [img.hullxy], -1, (0, 255, 0), thickness=1)
        img.help = cv2.bitwise_or(drawn_contours[:,:,0], drawn_contours[:,:,1])

        if a > Amin:
            print("{} > {} : {}".format(a, Amin, a/Amin))
            got_hull = True
            break
        else:
            print("{} < {} : {}".format(a, Amin, a/Amin))
            print("problema é area")

        Amin = max(A0, round(Amin - 0.03*img.area))
        Wc = min(W0, Wc + 0.5)

    drawn_contours = cv2.addWeighted(img.gray3ch, 0.5, drawn_contours, 0.7, 0)
    # save(img, "dilate", img.dilate)
    # save(img, "divide", img.divide)
    # save(img, "cannyG", img.cannyG)
    # save(img, "cannyS", img.cannyS)
    # save(img, "cannyV", img.cannyV)
    save(img, "cannyGSV", img.canny)
    save(img, "medgesforcontour", img.medges)
    save(img, "contours", drawn_contours)

    if not got_hull:
        print("finding board region failed")
        exit()

    return img

def find_morph(img):
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kd = 10
    kx = kd+round(kd/3)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (kd,kx))
    img.dilate = cv2.morphologyEx(img.claheG, cv2.MORPH_DILATE, k_dil)
    img.divide = cv2.divide(img.claheG, img.dilate, scale = 255)
    edges = cv2.threshold(img.divide, 0, 255, cv2.THRESH_OTSU)[1]
    edges = cv2.bitwise_not(edges)
    img.fedges = np.copy(edges)
    edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, ko, iterations = 1)
    edges = cv2.bitwise_or(edges, img.canny)
    # edges = cv2.bitwise_or(edges, img.help)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    img.cont = contours[np.argmax(areas)]
    img.hullxy = cv2.convexHull(img.cont)
    arclen = cv2.arcLength(img.hullxy,True)
    a = cv2.contourArea(img.hullxy)
    img.medges = edges

    return img, a

def find_canny(image, wmin = 6):
    c_thrl0 = 180
    c_thrh0 = 250
    c_thrl = c_thrl0
    c_thrh = c_thrh0
    if wmin < 7.5:
        clmin = 10
        ctmin = 30
    else:
        clmin = 30
        ctmin = 55

    while c_thrh > ctmin:
        canny = cv2.Canny(image, c_thrl, c_thrh)
        w = canny.mean()
        if w > wmin:
            print("{0:0=.2f} > {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
            break
        else:
            if w > random.uniform(0, w*2):
                print("{0:0=.2f} < {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
        c_thrl = max(clmin, c_thrl - 15)
        c_thrh = max(ctmin, c_thrh - 9)

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
        if lines is not None:
            if lines.shape[0] >= minlines:
                print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],th, h_thrv, h_minl, h_maxg))
                lines = radius_theta(lines)
                lines = filter_lines(img, lines)
                lines, angles = lines_kmeans(img, lines)
                print("angles: ", angles)
                got_hough = True
                break
            else:
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
            cv2.line(drawn_lines,(x1,y1),(x2,y2), (0,0,255), 3)
    img.select = drawn_lines[:,:,2]
    drawn_lines = cv2.addWeighted(img.hull3ch, 0.4, drawn_lines, 0.7, 0)
    save(img, "select", drawn_lines)

    img.select_lines = lines
    img.angles = angles
    img.slen = img.select_lines[:,0,4].min()
    return img

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
    img.hullBGR = cv2.resize(img.hullBGR, (img.hwidth, img.hheigth))
    img.medges = cv2.resize(img.medges, (img.hwidth, img.hheigth))
    img.G = cv2.resize(img.G, (img.hwidth, img.hheigth))
    img.gray = cv2.resize(img.gray, (img.hwidth, img.hheigth))
    img.claheG = cv2.resize(img.claheG, (img.hwidth, img.hheigth))
    # img.claheS = cv2.resize(img.claheS, (img.hwidth, img.hheigth))
    img.claheV = cv2.resize(img.claheV, (img.hwidth, img.hheigth))
    img.fedges = cv2.resize(img.fedges, (img.hwidth, img.hheigth))
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
    got_hough = False
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_DILATE, k_dil)
    save(img, "dilate", img.canny)
    img.canny = cv2.bitwise_and(img.canny, img.fedges)
    save(img, "and", img.canny)
    img.canny = cv2.bitwise_or(img.canny, img.select)
    save(img, "orselect", img.canny)
    img.canny = cv2.morphologyEx(img.canny, cv2.MORPH_CLOSE, k_dil)
    save(img, "close", img.canny)
    h_maxg0 = 200
    h_minl0 = round(img.slen)
    h_thrv0 = round(h_minl0 / 1.5)
    h_angl0 = np.pi / 1040

    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    minlines = 22
    while h_angl < (np.pi / 480):
        th = 180*(h_angl/np.pi)
        lines = cv2.HoughLinesP(img.canny, 1, h_angl,  h_thrv,  None, h_minl, h_maxg)
        if lines is not None:
            lines = radius_theta(lines)
            lines = filter_lines(img, lines)
            lines = filter_angles(img, lines)
            bundler = HoughBundler()
            lines = bundler.process_lines(lines)
            lines = radius_theta(lines)
            if lines.shape[0] >= minlines:
                print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],th, h_thrv, h_minl, h_maxg))
                got_hough = True
                break
            else:
                if th > random.uniform(0, th*4):
                    print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],th, h_thrv, h_minl, h_maxg))
        # if h_minl > h_minl0 / 2:
        #     h_minl -= 1
        #     h_thrv = round(h_minl / 1.2)

        h_angl += np.pi / 14400

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
    if not got_hough:
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
    diff.append((abs(centers[0] - 90), -85))
    diff.append((abs(centers[0] + 90), +85))
    diff.append((abs(centers[1] - 90), -85))
    diff.append((abs(centers[1] + 90), +85))
    if centers.shape[0] > 2:
        diff.append((abs(centers[2] - 90), -85))
        diff.append((abs(centers[2] + 90), +85))

    for d,k in diff:
        if d < 20:
            if abs(centers[0] - k) > 15 and abs(centers[1] - k) > 15:
                centers = np.append(centers, k)
            elif len(centers) > 2:
                if abs(centers[2] - k) > 15:
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

def pre_process(img):
    print("applying filter to image...")
    img.G = cv2.GaussianBlur(img.gray, (5,5), 1)
    # img.S = cv2.GaussianBlur(img.S)
    img.V = cv2.GaussianBlur(img.V, (5,5), 1)

    print("applying distributed histogram equalization to image...")
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    img.claheG = clahe.apply(img.G)
    # img.claheS = clahe.apply(img.S)
    img.claheV = clahe.apply(img.V)

    print("applying filter again...")
    img.claheG = lf.ffilter(img.claheG)
    # img.claheS = lf.ffilter(img.claheS)
    img.claheV = lf.ffilter(img.claheV)

    # save(img, "claheG", img.claheG)

    return img
