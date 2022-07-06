import cv2
from pathlib import Path

class Image:
    def __init__(self, filename):
        self.filename = filename
        self.basename = Path(self.filename).stem
        self.color = cv2.imread(self.filename)
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
