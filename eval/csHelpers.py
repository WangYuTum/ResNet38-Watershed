#!/usr/bin/python
#
# Various helper methods and includes for Cityscapes
#

# Python imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, getopt
import glob
import math
import json
from collections import namedtuple

# Image processing
# Check if PIL is actually Pillow as expected
try:
    from PIL import PILLOW_VERSION
except:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)

try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

# Numpy for datastructures
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

# Cityscapes modules
# try:
#     from annotation   import Annotation
#     from labels       import labels, name2label, id2label, trainId2label, category2labels
# except:
#     print("Failed to find all Cityscapes modules")
#     sys.exit(-1)

# Cityscapes labels
Label = namedtuple( 'Label' , [ 'name', 'id', 'trainId', 'category',
                                'categoryId', 'hasInstances', 'ignoreInEval', 'color' ] )
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       19 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
id2label = { label.id      : label for label in labels           }


# Print an error message and quit
def printError(message):
    print('ERROR: ' + str(message))
    sys.exit(-1)

# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val, args):
    if not args.colorized:
        return ""
    if not isinstance(val, float) or math.isnan(val):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

# Cityscapes files have a typical filename structure
# <city>_<sequenceNb>_<frameNb>_<type>[_<type2>].<ext>
# This class contains the individual elements as members
# For the sequence and frame number, the strings are returned, including leading zeros
CsFile = namedtuple( 'csFile' , [ 'city' , 'sequenceNb' , 'frameNb' , 'type' , 'type2' , 'ext' ] )

# Returns a CsFile object filled from the info in the given filename
def getCsFileInfo(fileName):
    baseName = os.path.basename(fileName)
    parts = baseName.split('_')
    parts = parts[:-1] + parts[-1].split('.')
    if not parts:
        printError( 'Cannot parse given filename ({}). Does not seem to be a valid Cityscapes file.'.format(fileName) )
    if len(parts) == 5:
        csFile = CsFile( *parts[:-1] , type2="" , ext=parts[-1] )
    elif len(parts) == 6:
        csFile = CsFile( *parts )
    else:
        printError( 'Found {} part(s) in given filename ({}). Expected 5 or 6.'.format(len(parts) , fileName) )

    return csFile

# Returns the part of Cityscapes filenames that is common to all data types
# e.g. for city_123456_123456_gtFine_polygons.json returns city_123456_123456
def getCoreImageFileName(filename):
    csFile = getCsFileInfo(filename)
    return "{}_{}_{}".format( csFile.city , csFile.sequenceNb , csFile.frameNb )

# Returns the directory name for the given filename, e.g.
# fileName = "/foo/bar/foobar.txt"
# return value is "bar"
# Not much error checking though
def getDirectory(fileName):
    dirName = os.path.dirname(fileName)
    return os.path.basename(dirName)

# Make sure that the given path exists
def ensurePath(path):
    if not path:
        return
    if not os.path.isdir(path):
        os.makedirs(path)

# Write a dictionary as json file
def writeDict2JSON(dictName, fileName):
    with open(fileName, 'w') as f:
        f.write(json.dumps(dictName, default=lambda o: o.__dict__, sort_keys=True, indent=4))

# dummy main
if __name__ == "__main__":
    printError("Only for include, not executable on its own.")
