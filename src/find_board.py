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

def update_hull(img):
    contours, _ = cv2.findContours(img.medges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cont = contours[max_index]
    hullxy = cv2.convexHull(cont)

    drawn_contours = np.empty(img.hull3ch.shape, dtype='uint8') * 0
    cv2.drawContours(drawn_contours, [hullxy], -1, (0, 255, 0), thickness=3)
    cv2.drawContours(drawn_contours, cont, -1, (255, 0, 0), thickness=3)
    drawn_contours = cv2.addWeighted(img.hull3ch, 0.5, drawn_contours, 0.8, 0)
    # save(img, "convex", drawn_contours)
    return hullxy

def find_board(img):
    # save(img, "sgray", img.sgray)
    img.wang0 = lwang.wang_filter(img.sgray)
    # save(img, "wang0", img.wang0)

    c = 3
    Amin = 0.45 * img.sarea
    increasing = True
    while c < 8:
        print("Amin =", Amin)
        print("clahe = ", c)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(c, c))
        img.clahe = clahe.apply(img.wang0)
        img.wang = lwang.wang_filter(img.clahe)
        img.canny = find_canny(img)
        img.medges,img.hullxy,img.got_hull,increasing = find_morph(img, Amin)
        if increasing:
            pass
            # save(img, "clahe@{}".format(c), img.clahe)
            # save(img, "wang@{}".format(c), img.wang)
            # save(img, "canny@{}".format(c), img.canny)

        if img.got_hull:
            break
        else:
            if not increasing:
                Amin -= 0.008 * img.sarea
                continue
            else:
                c += 1

    # save(img, "medges0", img.medges)
    drawn_contours = np.empty(img.gray3ch.shape, dtype='uint8') * 0
    cv2.drawContours(drawn_contours, [img.hullxy], -1, (0, 255, 0), thickness=3)
    drawn_contours = cv2.addWeighted(img.gray3ch, 0.5, drawn_contours, 0.8, 0)
    # save(img, "convex0", drawn_contours)
    limx, limy = broad_hull(img)

    img.medges = img.medges[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.wang = img.wang[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.clahe = img.clahe[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    limx[0] = round(limx[0] / img.sfact)
    limx[1] = round(limx[1] / img.sfact)
    limy[0] = round(limy[0] / img.sfact)
    limy[1] = round(limy[1] / img.sfact)

    img.hull = img.gray[limx[0]:limx[1]+1, limy[0]:limy[1]+1]
    img.hxoff = limx[0]
    img.hyoff = limy[0]
    img = reduce_hull(img)
    img.hull3ch = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR)

    img.canny = find_canny(img, wmin = 8)
    img.medges += img.canny
    save(img, "medges+canny", img.medges)
    img.angles, img.select_lines = find_angles(img)

    lines,inter = magic_lines(img)

    corners = find_corners(img, inter)

    return corners

def find_morph(img, Amin):
    got_hull = False
    alast = 0
    increasing = False
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kd = 5
    while kd <= 12:
        if (kd == 7):
            alast = a
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (kd,kd+round(kd/3)))
        dilate = cv2.morphologyEx(img.wang, cv2.MORPH_DILATE, k_dil)
        edges_gray = cv2.divide(img.wang, dilate, scale = 255)
        edges_bin = cv2.bitwise_not(cv2.threshold(edges_gray, 0, 255, cv2.THRESH_OTSU)[1])

        edges_bin = cv2.morphologyEx(edges_bin, cv2.MORPH_ERODE, ko, iterations = 1)
        edges_wcanny = edges_bin + img.canny
        contours, _ = cv2.findContours(edges_wcanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cont = contours[max_index]
        hullxy = cv2.convexHull(cont)
        a = cv2.contourArea(hullxy)
        if a > Amin:
            print("{} > {} @ ksize = {} [GOTHULL]".format(a, Amin, kd))
            got_hull = True
            break
        else:
            print("{} < {} @ ksize = {}".format(a, Amin, kd))
            kd += 1

    if not got_hull:
        diff = a - alast
        mdiff = 0.1 * Amin
        if (diff > mdiff) and (a > (img.sarea*0.15)):
            print("diff: {} > {}, increasing".format(diff, mdiff))
            increasing = True
        else:
            print("diff: {} < {}, not incr".format(diff, mdiff))
            increasing = False

    medges = edges_bin

    return medges, hullxy, got_hull, increasing

def broad_hull(img):
    Pxmin = img.hullxy[np.argmin(img.hullxy[:,0,0]),0]
    Pxmax = img.hullxy[np.argmax(img.hullxy[:,0,0]),0]
    Pymin = img.hullxy[np.argmin(img.hullxy[:,0,1]),0]
    Pymax = img.hullxy[np.argmax(img.hullxy[:,0,1]),0]

    Pxmin[0] = max(0, Pxmin[0]-20)
    Pymin[1] = max(0, Pymin[1]-40) # peças de trás vao além
    Pxmax[0] = min(img.swidth, Pxmax[0]+20)
    Pymax[1] = min(img.sheigth, Pymax[1]+20)

    return [Pymin[1],Pymax[1]], [Pxmin[0],Pxmax[0]]

def find_canny(img, wmin = 6):
    c_thrl0 = 80
    c_thrh0 = 200
    c_thrl = c_thrl0
    c_thrh = c_thrh0

    while c_thrh > 20:
        img.canny = cv2.Canny(img.wang, c_thrl, c_thrh)
        w = img.canny.mean()
        if w > wmin:
            print("{0:0=.2f} > {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
            break
        else:
            if wmin - w < wmin:
                print("{0:0=.2f} < {1}, @ {2}, {3}".format(w, wmin, c_thrl, c_thrh))
        if c_thrl > 10:
            c_thrl -= 9
        c_thrh -= 9

    return img.canny

def find_angles(img):
    got_hough = False
    h_maxg0 = 2
    h_minl0 = round((img.hwidth + img.hheigth)*0.2)
    h_thrv0 = round(h_minl0 / 10)
    h_angl0 = np.pi / 360

    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    minlines = 24
    while h_angl < (np.pi / 30):
        lines = cv2.HoughLinesP(img.canny, 1, h_angl,  h_thrv,  None, h_minl, h_maxg)
        if lines is not None and lines.shape[0] >= minlines:
            print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
            lines = radius_theta(lines)
            lines = filter_lines(img, lines)
            lines, angles = lines_kmeans(img, lines)
            print("angles: ", angles)
            got_hough = True
            break
        elif lines is not None:
            print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        if h_maxg < 20:
            h_maxg += 1
        if h_minl > h_minl0 / 2:
            h_minl -= 10
        if h_angl > (np.pi / 40):
            h_minl = 32
            h_maxg = 2
            minlines = 60

        h_thrv = round(h_minl / 10)
        h_angl += np.pi / 14400

    if not got_hough:
        lines = radius_theta(lines)
        lines = filter_lines(img, lines)
        lines, angles = lines_kmeans(img, lines)
        print("angles: ", angles)

    drawn_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            cv2.line(drawn_lines,(x1,y1),(x2,y2),(0,0,250),round(2/img.sfact))
    drawn_lines = cv2.addWeighted(img.hull3ch, 0.5, drawn_lines, 0.8, 0)
    save(img, "hough_select", drawn_lines)

    return angles, lines

def determinant(a, b):
    return a[0]*b[1] - a[1]*b[0]

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

            if abs(t - tt) < 30 or abs(t - tt) > 150:
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

            if img.got_hull and cv2.pointPolygonTest(img.shull, (x,y), True) < -20:
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
                    # print("Close point ignored: ({},{}) ~ ({},{})".format(last[0],last[1],x,y))
                    continue
        i += 1

    inter = np.array(inter, dtype='int32')
    return inter

def reduce_hull(img):
    img.hwidth = 900
    img.hfact = img.hwidth / img.hull.shape[1]
    img.hheigth = round(img.hfact * img.hull.shape[0])

    img.hull = cv2.resize(img.hull, (img.hwidth, img.hheigth))
    img.medges = cv2.resize(img.medges, (img.hwidth, img.hheigth))
    img.wang = cv2.resize(img.wang, (img.hwidth, img.hheigth))
    img.clahe = cv2.resize(img.clahe, (img.hwidth, img.hheigth))
    img.harea = img.hwidth * img.hheigth
    return img

def magic_lines(img):
    got_hough = False
    h_minl0 = round((img.hwidth + img.hheigth)*0.2)
    h_thrv0 = round(h_minl0 / 1.5)
    h_maxg0 = round(h_minl0 / 50) + 5
    h_angl0 = np.pi / 1440

    h_maxg = h_maxg0
    h_minl = h_minl0
    h_thrv = h_thrv0
    h_angl = h_angl0
    j = 0
    while h_angl < (np.pi / 90):
        lines = cv2.HoughLinesP(img.medges, 1, h_angl, h_thrv,  None, h_minl, h_maxg)
        if lines is not None:
            lines = radius_theta(lines)
            lines = filter_lines(img, lines)
            lines = filter_angles(img, lines)
            linesbef = np.copy(lines)
            bundler = HoughBundler()
            lines = bundler.process_lines(lines)
            lines = radius_theta(lines)
            if lines.shape[0] >= 10:
                print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
                aux = np.copy(img.select_lines[:,:,0:6])
                lines = np.append(lines, aux, axis=0)
                got_hough = True
                break

        if lines is not None:
            print("{0} lines @ {1:1=.4f}º, {2}, {3}, {4}".format(lines.shape[0],180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))
        if h_minl > h_minl0 / 5:
            h_minl -= 3
            h_thrv = round(h_minl / 1.5)
            h_maxg = round(h_minl / 50) + 5
        j += 1
        h_angl += np.pi / 14400

    if got_hough:
        drawn_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        draw_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        for line in linesbef:
            for x1,y1,x2,y2,r,t in line:
                cv2.line(draw_lines,(x1,y1),(x2,y2),(0,0,255),round(2/img.sfact))
        drawn_lines = cv2.addWeighted(img.hull3ch, 0.5, draw_lines, 0.8, 0)
        save(img, "hough_bef_mer", drawn_lines)

        drawn_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        draw_lines = cv2.cvtColor(img.hull, cv2.COLOR_GRAY2BGR) * 0
        for line in lines:
            for x1,y1,x2,y2,r,t in line:
                cv2.line(draw_lines,(x1,y1),(x2,y2),(0,0,255),round(2/img.sfact))
        drawn_lines = cv2.addWeighted(img.hull3ch, 0.5, draw_lines, 0.8, 0)
        save(img, "hough_aft_mer", drawn_lines)

        draw_lines = draw_lines[:,:,2]
        img.medges += draw_lines

        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        img.medges = cv2.morphologyEx(img.medges, cv2.MORPH_CLOSE, ko, iterations = 1)

        # save(img, "medgesCLOSED", img.medges)
        img.shull = update_hull(img)
        inter = find_intersections(img, lines[:,0,:])

        drawn_circles = np.copy(img.hull3ch) * 0
        for p in inter:
            cv2.circle(drawn_circles, p, radius=7, color=(255, 0, 0), thickness=-1)
        drawn_circles = cv2.addWeighted(img.hull3ch, 0.5, drawn_circles, 0.8, 0)
        save(img, "intersections".format(img.basename), drawn_circles)
    else:
        print("FAILED @ {}, {}, {}, {}".format(180*(h_angl/np.pi), h_thrv, h_minl, h_maxg))

    return lines,inter

def filter_lines(img, lines):
    """
    remove lines that are on the border of the image and are horizontal or vertical
    """
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

def filter_angles(img, lines):
    rem = np.empty(lines.shape[0])
    rem = np.int32(rem)

    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            if abs(t - img.angles[0]) > 10 and abs(t - img.angles[1]) > 10:
                if img.angles.shape[0] == 2:
                    rem[i] = 1
                elif abs(t - img.angles[2]) > 10:
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

    fig = plt.figure()
    plt.xticks(range(-90, 91, 10))
    plt.hist(A[:,5], 180, [-90, 90], color = (0.9, 0.0, 0.0, 0.9))
    plt.hist(B[:,5], 180, [-90, 90], color = (0.0, 0.0, 0.9, 0.9))
    plt.hist(C[:,5], 180, [-90, 90], color = (0.0, 0.9, 0.0, 0.9))
    plt.hist(centers, 20, [-90, 90], color = (0.7, 0.7, 0.0, 0.8))
    savefig(img, "kmeans0", fig)

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
        fig = plt.figure()
        plt.xticks(range(-90, 91, 10))
        plt.hist(A[:,5], 180, [-90, 90], color = (0.9, 0.0, 0.0, 0.9))
        plt.hist(B[:,5], 180, [-90, 90], color = (0.0, 0.0, 0.9, 0.9))
        plt.hist(centers, 20, [-90, 90], color = (0.7, 0.7, 0.0, 0.7))
        savefig(img, "kmeans1", fig)

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
            centers = np.append(centers, k)
            break

    lines = np.int32(lines)
    return lines, centers

def find_corners(img, inter):
    print("inter:", inter.shape)

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

    print("points:", BR, BL, TR, TL)

    drawn_circles = np.copy(img.hull3ch) * 0
    for p in BR, BL, TR, TL:
        cv2.circle(drawn_circles, p, radius=7, color=(255, 0, 0), thickness=-1)
    drawn_circles = cv2.addWeighted(img.hull3ch, 0.5, drawn_circles, 0.8, 0)
    save(img, "corners".format(img.basename), drawn_circles)

    return BR, BL, TR, TL
