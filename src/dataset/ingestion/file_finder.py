import glob
import os

class FileFinder:
    @classmethod
    def find_files(cls, dir_path, extension, limit=-1):
        glob_str = os.path.join(dir_path, "*"+extension)
        return glob.glob(glob_str)[:limit]
