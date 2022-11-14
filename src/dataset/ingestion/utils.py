import re


def remove_alpha(strng):
    return int(re.sub('[^0-9]','', strng))