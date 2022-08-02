import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np

i = 1

def logprint(img, text):
    if img.log:
        print(text)

def determinant(a, b):
    return a[0]*b[1] - a[1]*b[0]

def save(img, filename, image):
    global i
    cv2.imwrite("{}{:02d}_{}.png".format(img.basename, i, filename), image)
    i += 1

def savefig(img, filename, fig):
    global i
    fig.savefig("{}{:02d}_{}.png".format(img.basename, i, filename))
    i += 1

def radius(x1,y1,x2,y2):
    dx = x2 - x1
    dy = y2 - y1
    radius = math.sqrt(dx*dx + dy*dy)
    return round(radius)

def theta(x1,y1,x2,y2):
    orientation = math.atan2(y1-y2, x2-x1)
    orientation = math.degrees(orientation)
    return round(orientation)

def radius_theta(lines):
    dummy = np.zeros((lines.shape[0], 1, 6), dtype='int32')
    dummy[:,0,0:4] = np.copy(lines[:,0,0:4])
    lines = dummy
    i = 0
    for line in lines:
        for x1,y1,x2,y2,r,t in line:
            lines[i, 0, 4] = radius(x1,y1,x2,y2)
            lines[i, 0, 5] = theta(x1,y1,x2,y2)
            i += 1
    return lines

def geo_lines(img, lines):
    lines = radius_theta(lines)
    vert = lines[abs(lines[:,0,5]) > 45]
    hori = lines[abs(lines[:,0,5]) < 45]

    vert = vert[vert[:,0,0].argsort()]
    hori = hori[hori[:,0,1].argsort()]

    return vert, hori

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
