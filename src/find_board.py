import cv2
import numpy as np
import math
import sys
from Image import Image
from pathlib import Path
from os.path import exists

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import lwang

def save(img, filename, image):
    if img.save and not exists(filename):
        cv2.imwrite(filename, image)

def shortest_connections(img, intersections):
    drawn_lines = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0

    for x1, y1 in intersections:
        distance = 0
        seconx = []
        secony = []
        distxy = []
        anglxy = []

        for x2, y2 in intersections:
            if (x1, y1) == (x2, y2):
                continue
            else:
                dista = radius(x1,y1,x2,y2)
                angle = theta(x1,y1,x2,y2)
                if dista > 10:
                    seconx.append(x2)
                    secony.append(y2)
                    distxy.append(dista)
                    anglxy.append(angle)
                else:
                    continue

        connections = list(zip(distxy, anglxy, seconx, secony))
        connections = np.array(connections)
        connections = connections[connections[:,0].argsort()]

        for c in connections[:2]:
            neg = (round(c[2]), round(c[3]))
            if c[0] < 50:
                cv2.line(drawn_lines, (x1,y1), (neg[0], neg[1]), (0,0,255), round(2/img.fact))
            else:
                continue

    return drawn_lines

def det(a, b):
    return a[0]*b[1] - a[1]*b[0]

def find_intersections(img, lines):
    inter = []
    last = (0,0)

    i = 0
    for x1, y1, x2, y2, r, t in lines:
        l1 = [(x1,y1), (x2,y2)]
        j = 0
        for xx1, yy1, xx2, yy2, rr, tt in lines:
            l2 =  [(xx1,yy1), (xx2,yy2)]
            if (x1,y1) == (xx1,yy1) and (x2,y2) == (xx2,yy2):
                continue

            if abs(t - tt) < 30:
                continue

            xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
            ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

            div = det(xdiff, ydiff)
            if div == 0:
                j += 1
                continue

            d = (det(*l1), det(*l2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div

            if x > img.swidth or y > img.sheigth or x < 0 or y < 0:
                j += 1
                continue
            else:
                j += 1
                if radius(last[0], last[1], x, y) > 10:
                    inter.append((x,y))
                    last = (x,y)
                else:
                    print("Close point ignored: ({},{}) ~ ({},{})".format(last[0],last[1],x,y))
                    continue
        i += 1

    inter = np.array(inter, dtype='int32')
    return inter

def radius(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def theta(x1,y1,x2,y2):
    return math.degrees(math.atan2((y2-y1),(x2-x1)))

def draw_hough(img, lines):
    drawn_lines = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(drawn_lines,(x1,y1),(x2,y2),(0,0,250),round(2/img.fact))

    return drawn_lines

def find_hull(img):
    img_wang = lwang.wang_filter(img.small)
    save(img, "0{}_1wang.png".format(img.basename), img_wang)

    got = False
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    k_dil_s = 5
    lasta = 0
    while k_dil_s <= 30:
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dil_s,k_dil_s+3))
        dilate = cv2.morphologyEx(img_wang, cv2.MORPH_DILATE, k_dil)
        edges_gray = cv2.divide(img_wang, dilate, scale = 255)
        edges_bin = cv2.bitwise_not(cv2.threshold(edges_gray, 0, 255, cv2.THRESH_OTSU)[1])
        edges_opened = cv2.morphologyEx(edges_bin, cv2.MORPH_OPEN, k_open, iterations = 1)
        contours, _ = cv2.findContours(edges_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        perim = [cv2.arcLength(c, True) for c in contours]
        max_index = np.argmax(areas)
        a = areas[max_index]
        if a > (0.25 * img.sarea):
            print("{} > {}, p = {}".format(a, img.sarea, perim[max_index]))
            got = True
            break
        elif a > lasta - 20000:
            print("{} < {}, p = {}".format(a, img.sarea, perim[max_index]))
            k_dil_s += 1
            lasta = a
        else:
            print("{} < {}: failed. p = {}".format(a, img.sarea, perim[max_index]))
            break

    img_contour = np.empty(img.gray3ch.shape, dtype='uint8') * 0
    cont = contours[max_index]
    hull = cv2.convexHull(cont)
    cv2.drawContours(img_contour, [hull], -1, (0, 240, 0), thickness=3)
    cv2.drawContours(img_contour, cont, -1, (255,0,0), thickness=3)

    # shape = cv2.approxPolyDP(cont, 120, True)
    # cv2.drawContours(img_contour, [shape], -1, (0, 0, 250), thickness=3)

    img_contour_drawn = cv2.addWeighted(img.gray3ch, 0.5, img_contour, 0.8, 0)

    save(img, "0{}_3dilate.png".format(img.basename),     dilate)
    save(img, "0{}_4edges_gray.png".format(img.basename), edges_gray)
    save(img, "0{}_5edges_bin.png".format(img.basename),  edges_bin)
    save(img, "0{}_6edges_opened.png".format(img.basename),     edges_opened)
    save(img, "0{}_7countours.png".format(img.basename),  img_contour_drawn)

    return hull


def find_lines(img, c_thl, c_thh, h_th, h_minl, h_maxg):
    img_contour_bin = img_contour[:,:,0]
    lines = cv2.HoughLinesP(img_contour_bin, 2, np.pi / 180,  h_thrv,  None, h_minl, h_maxg)
    drawn_lines = draw_hough(img, lines)
    img_hough = cv2.addWeighted(img.gray3ch, 0.5, drawn_lines, 0.8, 0)
    if img.save:
        cv2.imwrite("0{}_8edges{}_{}_hough{}_{}_{}.png".format(img.basename, 8, 8, h_thrv, h_minl, h_maxg), img_hough)

    aux = np.zeros((lines.shape[0], 1, 6))
    aux[:,0,0:4] = np.copy(lines[:,0,0:4])
    lines = np.float32(aux)
    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            lines[i, 0, 4] = radius(x1,y1,x2,y2)
            lines[i, 0, 5] = theta(x1,y1,x2,y2)
            i += 1
    return lines

def broad_hull(img, hull):
    Pxmin = hull[np.argmin(hull[:,0,0]),0]
    Pxmax = hull[np.argmax(hull[:,0,0]),0]
    Pymin = hull[np.argmin(hull[:,0,1]),0]
    Pymax = hull[np.argmax(hull[:,0,1]),0]

    Pxmin[0] = max(0, Pxmin[0]-10)
    Pymin[1] = max(0, Pymin[1]-10)
    Pxmax[0] = min(img.swidth, Pxmax[0]+10)
    Pymax[1] = min(img.sheigth, Pymax[1]+10)

    if Pxmin[1] < Pxmax[1]:
        Pxmin[1] -= 10
        Pxmax[1] += 10
    else:
        Pxmax[1] -= 10
        Pxmin[1] += 10

    if Pymin[0] < Pymax[0]:
        Pymin[0] -= 10
        Pymax[0] += 10
    else:
        Pymax[0] -= 10
        Pymin[0] += 10
    return (Pymin[1],Pymax[1]), (Pxmin[0],Pxmax[0])

def find_board(img, c_thrl, c_thrh, h_thrv, h_minl, h_maxg):
    if img.save:
        cv2.imwrite("0{}_0gray.png".format(img.basename, c_thrl, c_thrh), img.small)

    hull = find_hull(img)
    limx, limy = broad_hull(img, hull)
    img.hull = img.small[limx[0]:limx[1], limy[0]:limy[1]]

    save(img, "0{}_9cuthull.png".format(img.basename), img.hull)

    exit()
    # lines = find_lines(img, c_thl, c_thh, h_th, h_minl, h_maxg)

    intersections = find_intersections(img, lines[:,0,:])
    intersections = intersections[intersections[:,0].argsort()]
    intersections = np.unique(intersections, axis=0)
    intersections = intersections[intersections[:,0].argsort()]

    drawn_circles = cv2.cvtColor(img.small, cv2.COLOR_GRAY2BGR) * 0

    for p in intersections:
        cv2.circle(drawn_circles, p, radius=3, color=(255, 0, 0), thickness=-1)

    points = drawn_circles[:,:,0]
    image = cv2.addWeighted(img.gray3ch, 0.5, drawn_circles, 0.8, 0)

    save(img, "0{}_9intersections.png".format(img.basename), image)

    # drawn_lines = shortest_connections(img, intersections)
    # conn = cv2.addWeighted(img.gray3ch, 0.5, drawn_lines, 0.8, 0)

    return (10, 300, 110, 310)
