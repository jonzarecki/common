import sys
import codecs


class Tee:
    """
    helper object, helps redirect the standard output to the screen and to a specified file
    """
    def __init__(self, fname, mode="w+"):
        self.file = codecs.open(fname, mode, 'utf8', errors="ignore")

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __del__(self):
        self.__exit__(None, None, None)
