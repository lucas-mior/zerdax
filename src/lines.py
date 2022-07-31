import math
import numpy as np

class HoughBundler:     
    def __init__(self,min_dist=8,min_angle=20):
        self.min_dist = min_dist
        self.min_angle = min_angle
    
    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_dist_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_dist(line_2, line_1) < min_dist_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def dist_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line[0:4]

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            dist_point_to_line = 9999
            return dist_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter dist
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                dist_point_to_line = iy
            else:
                dist_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            dist_point_to_line = line_magnitude(px, py, ix, iy)

        return dist_point_to_line

    def get_dist(self, a_line, b_line):
        dist1 = self.dist_point_to_line(a_line[0:2], b_line)
        dist2 = self.dist_point_to_line(a_line[2:4], b_line)
        dist3 = self.dist_point_to_line(b_line[0:2], a_line)
        dist4 = self.dist_point_to_line(b_line[2:4], a_line)


        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_dist, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        if(len(lines) == 1):
            return np.block([[lines[0][0:2], lines[0][2:4]]])

        lines = np.copy(lines)
        dummy = np.zeros((lines.shape[0], 5), dtype='int32')
        dummy[:,0:4] = np.copy(lines[:,0:4])
        lines = dummy
        i = 0
        for line in lines:
            lines[i,4] = self.get_orientation(line)
            i += 1

        orien = np.median(lines[:,4])
        for line in lines:
            if abs(line[4] - orien) <= 2:
                a = [[line[0:2], line[2:4]]]
                break

        return np.block(a)

        #points = []
        #for line in lines:
        #    points.append(line[0:2])
        #    points.append(line[2:4])
        #if 45 < orientation <= 90:
        #    #sort by y
        #    points = sorted(points, key=lambda point: point[1])
        #else:
        #    #sort by x
        #    points = sorted(points, key=lambda point: point[0])

        #return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []
  
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)
                    
        return np.asarray(merged_lines_all)
