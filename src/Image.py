from pathlib import Path

class Image:
    def __init__(self, filename):
        self.filename = filename
        self.basename = Path(self.filename).stem
