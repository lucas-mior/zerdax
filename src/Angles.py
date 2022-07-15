import numpy as np
class Angles:
    def set(angles):
        a0 = np.array([
              [0, 0, 0, 0, 0,],
              [0, 0, 0, 0, 0,],
              [1, 1, 1, 1, 1,],
              [0, 0, 0, 0, 0,],
              [0, 0, 0, 0, 0,],
              ], dtype='uint8').T
        a2 = np.array([
              [0, 0, 0, 0, 1,],
              [0, 0, 0, 1, 1,],
              [0, 0, 1, 0, 0,],
              [1, 1, 0, 0, 0,],
              [1, 0, 0, 0, 0,],
              ], dtype='uint8').T
        a4 = np.array([
              [0, 0, 0, 0, 1,],
              [0, 0, 0, 1, 0,],
              [0, 0, 1, 0, 0,],
              [0, 1, 0, 0, 0,],
              [1, 0, 0, 0, 0,],
              ], dtype='uint8').T
        a6 = np.array([
              [0, 0, 0, 1, 1,],
              [0, 0, 0, 1, 0,],
              [0, 0, 1, 0, 0,],
              [0, 1, 0, 0, 0,],
              [1, 1, 0, 0, 0,],
              ], dtype='uint8').T
        a9 = np.array([
              [0, 0, 1, 0, 0,],
              [0, 0, 1, 0, 0,],
              [0, 0, 1, 0, 0,],
              [0, 0, 1, 0, 0,],
              [0, 0, 1, 0, 0,],
              ], dtype='uint8').T

        i = 0
        k0 = np.zeros((angles.shape[0], 5, 5), dtype='uint8')
        k = np.zeros((angles.shape[0], 5, 5), dtype='uint8')
        for angle in angles:
            if abs(angle) - 0.0) <= 11.25:
                k0[i,:,:] = a0
            elif abs(angle - 22.5) <= 11.25:
                k0[i,:,:] = a2
            elif abs(angle - 45.0) <= 11.25:
                k0[i,:,:] = a4
            elif abs(angle - 67.5) <= 11.25:
                k0[i,:,:] = a6
            elif abs(angle - 90) <= 11.25:
                k0[i,:,:] = a9
            else:
                k0[i,:,:] = a9
            i += 1

        i -= 1
        if i == 2:
            k = k0[0] + k0[1]
        elif i == 3:
            k = k0[0] + k0[1] + k0[2]
        return k
