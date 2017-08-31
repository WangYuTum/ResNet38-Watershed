'''
The script is to evaluate pixel level semantic labeling.
Support scripts needed:
 - addToConfusionMatrix.pyx
 - addToConfusionMatrix_impl.c
 - addToConfusionMatrix.c
 - csHelpers.py
 - setup.py

 NOTE: enable cython before running evaluation(already enabled on server):
 setup.py build_ext --inplace

 NOTE:
  - specify environment variables before evaluation
  CITYSCAPES_DATASET, CITYSCAPES_RESULTS

 USAGE:
  - Put all ground truth files (validation groundtruth - <city>_123456_123456_gtFine_labelIds.png ) in :
    os.environ['CITYSCAPES_GROUNDTRUTH']
  - Put all prediction files (validation predictions - <city>_123456_123456*.png ) in:
    os.environ['CITYSCAPES_RESULTS']
  - The number of prediction files and number of groundtruth MUST be the same.
  - The evaluation method reads prediction results in os.environ['CITYSCAPES_RESULTS']
    and groundtruth files in os.environ['CITYSCAPES_GROUNDTRUTH'] to Calculate accuracy.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import platform
import fnmatch
from PIL import Image
try:
    from itertools import izip
except ImportError:
    izip = zip

from eval.csHelpers import *

CSUPPORT = True
if CSUPPORT:
    try:
        import addToConfusionMatrix
    except:
        CSUPPORT = False

os.environ['CITYSCAPES_DATASET'] = '../data/CityDatabase'
# To specify where the predictions are
os.environ['CITYSCAPES_RESULTS'] = '../data/test_city_labelIDs'
# To specify where the ground truth are
os.environ['CITYSCAPES_GROUNDTRUTH'] = '../data/test_city_valGT'


###################
# Global parameters
###################

class CArgs(object):
	pass
args = CArgs()

# Specify database path
if 'CITYSCAPES_DATASET' in os.environ:
	args.cityscapesPath = os.environ['CITYSCAPES_DATASET']
else:
	print('No dataset path specified, use default path: {}'.format('../data/CityDatabase'))
	args.cityscapesPath = '../data/CityDatabase'

if 'CITYSCAPES_GROUNDTRUTH' in os.environ:
	args.groundTruthSearch = os.path.join(os.environ['CITYSCAPES_GROUNDTRUTH'], "*_gtFine_labelIds.png")
else:
	args.groundTruthSearch  = os.path.join( args.cityscapesPath , "gtFine" , "val" , "*", "*_gtFine_labelIds.png" )


args.evalInstLevelScore = False
args.evalPixelAccuracy  = True
args.evalLabels         = []
args.printRow           = 5
args.normalized         = True
args.colorized          = hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and platform.system()=='Linux'
args.bold               = colors.BOLD if args.colorized else ""
args.nocol              = colors.ENDC if args.colorized else ""
args.JSONOutput         = True
args.quiet              = False
args.debug				= False

args.avgClassSize       = {
    "bicycle"    :  4672.3249222261 ,
    "caravan"    : 36771.8241758242 ,
    "motorcycle" :  6298.7200839748 ,
    "rider"      :  3930.4788056518 ,
    "bus"        : 35732.1511111111 ,
    "train"      : 67583.7075812274 ,
    "car"        : 12794.0202738185 ,
    "person"     :  3462.4756337644 ,
    "truck"      : 27855.1264367816 ,
    "trailer"    : 16926.9763313609 ,
}

args.predictionPath = None
args.predictionWalk = None

# Get prediction for the given groundtruth file
# NOTE: specify prediction file in an environment variable CITYSCAPES_RESULTS
# The prediction file MUST have the following pattern:
# <city>_123456_123456*.png
# the respective groundtruth file has a name:
# <city>_123456_123456_gtFine_labelIds.png
def getPrediction( args, groundTruthFile ):
    # determine the prediction path, if the method is first called
    if not args.predictionPath:
        rootPath = None
        if 'CITYSCAPES_RESULTS' in os.environ:
            rootPath = os.environ['CITYSCAPES_RESULTS']
        if not os.path.isdir(rootPath):
            printError("Could not find a prediction folder.")

        args.predictionPath = rootPath

    # walk the prediction path, if not happened yet
    if not args.predictionWalk:
        walk = []
        for root, dirnames, filenames in os.walk(args.predictionPath):
            walk.append( (root,filenames) )
        args.predictionWalk = walk

    csFile = getCsFileInfo(groundTruthFile)
    filePattern = "{}_{}_{}*.png".format( csFile.city , csFile.sequenceNb , csFile.frameNb )

    predictionFile = None
    for root, filenames in args.predictionWalk:
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)
            else:
                printError("Found multiple predictions for ground truth {}".format(groundTruthFile))

    if not predictionFile:
        printError("Found no prediction for ground truth {}".format(groundTruthFile))

    if args.debug:
		print("Got the ground truth file: %s"%groundTruthFile)
		print("Got the prediction file: %s"%predictionFile)

    return predictionFile

# Generate empty confusion matrix and create list of relevant labels
def generateMatrix(args):
    args.evalLabels = []
    for label in labels:
        if (label.id < 0):
            continue
        # we append all found labels, regardless of being ignored
        args.evalLabels.append(label.id)
    maxId = max(args.evalLabels)
    # We use longlong type to be sure that there are no overflows
    return np.zeros(shape=(maxId+1, maxId+1),dtype=np.ulonglong)

'''
def generateInstanceStats(args):
    instanceStats = {}
    instanceStats["classes"   ] = {}
    instanceStats["categories"] = {}
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            instanceStats["classes"][label.name] = {}
            instanceStats["classes"][label.name]["tp"] = 0.0
            instanceStats["classes"][label.name]["tpWeighted"] = 0.0
            instanceStats["classes"][label.name]["fn"] = 0.0
            instanceStats["classes"][label.name]["fnWeighted"] = 0.0
    for category in category2labels:
        labelIds = []
        allInstances = True
        for label in category2labels[category]:
            if label.id < 0:
                continue
            if not label.hasInstances:
                allInstances = False
                break
            labelIds.append(label.id)
        if not allInstances:
            continue

        instanceStats["categories"][category] = {}
        instanceStats["categories"][category]["tp"] = 0.0
        instanceStats["categories"][category]["tpWeighted"] = 0.0
        instanceStats["categories"][category]["fn"] = 0.0
        instanceStats["categories"][category]["fnWeighted"] = 0.0
        instanceStats["categories"][category]["labelIds"] = labelIds

    return instanceStats
'''

# Calculate and return IOU score for a particular label
def getIouScoreForLabel(label, confMatrix, args):
    if id2label[label].ignoreInEval:
        if args.debug:
            print('label %s is ignored.'%label)
        return float('nan')

    # the number of true positive pixels for this label
    # the entry on the diagonal of the confusion matrix
    tp = np.longlong(confMatrix[label,label])

    # the number of false negative pixels for this label
    # the row sum of the matching row in the confusion matrix
    # minus the diagonal entry
    fn = np.longlong(confMatrix[label,:].sum()) - tp

    # the number of false positive pixels for this labels
    # Only pixels that are not on a pixel with ground truth label that is ignored
    # The column sum of the corresponding column in the confusion matrix
    # without the ignored rows and without the actual label of interest
    notIgnored = [l for l in args.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        if args.debug:
            print('label %s denom is 0.'%label)
        return float('nan')

    # return IOU
    return float(tp) / denom

'''
# Calculate and return iIOU score for a particular label
def getInstanceIouScoreForLabel(label, confMatrix, instStats, args):
    if id2label[label].ignoreInEval:
        return float('nan')

    labelName = id2label[label].name
    if not labelName in instStats["classes"]:
        return float('nan')

    tp = instStats["classes"][labelName]["tpWeighted"]
    fn = instStats["classes"][labelName]["fnWeighted"]
    # false postives computed as above
    notIgnored = [l for l in args.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom
'''

# Calculate prior for a particular class id. Used to generate result dictionary
def getPrior(label, confMatrix):
    return float(confMatrix[label,:].sum()) / confMatrix.sum()

# Get average of scores.
# Only computes the average over valid entries.
def getScoreAverage(scoreList, args):
    validScores = 0
    scoreSum    = 0.0
    for score in scoreList:
        if not math.isnan(scoreList[score]):
            validScores += 1
            scoreSum += scoreList[score]
    if validScores == 0:
        print('getAverage: valid score is 0.')
        return float('nan')
    return scoreSum / validScores

# Calculate and return IOU score for a particular category
def getIouScoreForCategory(category, confMatrix, args):
	'''
	No need for now
	'''
	pass

# Calculate and return IOU score for a particular category
def getInstanceIouScoreForCategory(category, confMatrix, instStats, args):
	'''
	No need for now
	'''
	pass

# create a dictionary containing all relevant results
def createResultDict( confMatrix, classScores, classInstScores, categoryScores, categoryInstScores, perImageStats, args ):
	'''
	Implement later. Not crucial
	'''
	pass

# Write results to a json file.
def writeJSONFile(wholeData, args):
	'''
	Implement later. Not crucial
	'''
	pass

# Evaluate image lists pairwise.
def evaluateImgLists(predictionImgList, groundTruthImgList, args):
	'''
	Evaluate for each pair of prediction and groundTruth in the respective lists.
	'''
	if len(predictionImgList) != len(groundTruthImgList):
		printError("List of images for prediction and groundtruth are not of equal size.")
	confMatrix    = generateMatrix(args)
	instStats = None
	perImageStats = {}
	nbPixels      = 0
	if args.evalInstLevelScore:
		instStats = generateInstanceStats(args)

	if not args.quiet:
		print("Evaluating {} pairs of images...".format(len(predictionImgList)))

    # Evaluate all pairs of images and save them into a matrix
	for i in range(len(predictionImgList)):
		predictionImgFileName = predictionImgList[i]
		groundTruthImgFileName = groundTruthImgList[i]
		#print "Evaluate ", predictionImgFileName, "<>", groundTruthImgFileName
		nbPixels += evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, instStats, perImageStats, args)

		# sanity check
		if confMatrix.sum() != nbPixels:
			printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

		if not args.quiet:
			print("\rImages Processed: {}".format(i+1), end=' ')
			sys.stdout.flush()
	if not args.quiet:
		print("\n")

    # sanity check
	if confMatrix.sum() != nbPixels:
		printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

    # print confusion matrix
	'''
	if (not args.quiet):
        printConfMatrix(confMatrix, args)
	'''

    # Calculate IOU scores on class level from matrix
	classScoreList = {}
	for label in args.evalLabels:
		labelName = id2label[label].name
		classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, args)

	if args.debug:
		for labelname in classScoreList:
			print('Score of {}: {}'.format(labelname,classScoreList[labelname]))

	avgScore = getScoreAverage(classScoreList, args)
	print('The average score is {}'.format(avgScore))

    # Calculate instance IOU scores on class level from matrix
	'''
	classInstScoreList = {}
	for label in args.evalLabels:
        labelName = id2label[label].name
        classInstScoreList[labelName] = getInstanceIouScoreForLabel(label, confMatrix, instStats, args)
	'''


    # Print IOU scores
	'''
	if (not args.quiet):
        print("")
        print("")
        printClassScores(classScoreList, classInstScoreList, args)
        iouAvgStr  = getColorEntry(getScoreAverage(classScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(classScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(classInstScoreList , args), args) + "{avg:5.3f}".format(avg=getScoreAverage(classInstScoreList , args)) + args.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")
	'''


    # Calculate IOU scores on category level from matrix
	'''
	categoryScoreList = {}
	for category in category2labels.keys():
        categoryScoreList[category] = getIouScoreForCategory(category,confMatrix,args)

	# Calculate instance IOU scores on category level from matrix
	categoryInstScoreList = {}
	for category in category2labels.keys():
        categoryInstScoreList[category] = getInstanceIouScoreForCategory(category,confMatrix,instStats,args)

	# Print IOU scores
	if (not args.quiet):
        print("")
        printCategoryScores(categoryScoreList, categoryInstScoreList, args)
        iouAvgStr = getColorEntry(getScoreAverage(categoryScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(categoryScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(categoryInstScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(categoryInstScoreList, args)) + args.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")
	'''

    # write result file
	'''
	allResultsDict = createResultDict( confMatrix, classScoreList, classInstScoreList, categoryScoreList, categoryInstScoreList, perImageStats, args )
	writeJSONFile( allResultsDict, args)
	'''

    # return confusion matrix
    # return allResultsDict
	return avgScore

# Main evaluation method. Evaluates pairs of prediction and ground truth
# images which are passed as arguments.
def evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, instanceStats, perImageStats, args):
    # Loading all resources for evaluation.
	try:
		predictionImg = Image.open(predictionImgFileName)
		predictionNp  = np.array(predictionImg)
	except:
		printError("Unable to load " + predictionImgFileName)
	try:
		groundTruthImg = Image.open(groundTruthImgFileName)
		groundTruthNp = np.array(groundTruthImg)
	except:
		printError("Unable to load " + groundTruthImgFileName)

    # load ground truth instances, if needed. False anyway for now.
	'''
	if args.evalInstLevelScore:
        groundTruthInstanceImgFileName = groundTruthImgFileName.replace("labelIds","instanceIds")
        try:
            instanceImg = Image.open(groundTruthInstanceImgFileName)
            instanceNp  = np.array(instanceImg)
        except:
            printError("Unable to load " + groundTruthInstanceImgFileName)
	'''

    # Check for equal image sizes
	if (predictionImg.size[0] != groundTruthImg.size[0]):
		printError("Image widths of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
	if (predictionImg.size[1] != groundTruthImg.size[1]):
		printError("Image heights of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
	if ( len(predictionNp.shape) != 2 ):
		printError("Predicted image has multiple channels.")

	imgWidth  = predictionImg.size[0]
	imgHeight = predictionImg.size[1]
	nbPixels  = imgWidth*imgHeight

    # Evaluate images
	if (CSUPPORT):
		# using cython
		confMatrix = addToConfusionMatrix.cEvaluatePair(predictionNp, groundTruthNp, confMatrix, args.evalLabels)
	else:
		# the slower python way
		for (groundTruthImgPixel,predictionImgPixel) in izip(groundTruthImg.getdata(),predictionImg.getdata()):
			if (not groundTruthImgPixel in args.evalLabels):
				printError("Unknown label with id {:}".format(groundTruthImgPixel))

			confMatrix[groundTruthImgPixel][predictionImgPixel] += 1

	if args.evalInstLevelScore:
        # Generate category masks
        categoryMasks = {}
        for category in instanceStats["categories"]:
            categoryMasks[category] = np.in1d( predictionNp , instanceStats["categories"][category]["labelIds"] ).reshape(predictionNp.shape)

        instList = np.unique(instanceNp[instanceNp > 1000])
        for instId in instList:
            labelId = int(instId/1000)
            label = id2label[ labelId ]
            if label.ignoreInEval:
                continue

            mask = instanceNp==instId
            instSize = np.count_nonzero( mask )

            tp = np.count_nonzero( predictionNp[mask] == labelId )
            fn = instSize - tp

            weight = args.avgClassSize[label.name] / float(instSize)
            tpWeighted = float(tp) * weight
            fnWeighted = float(fn) * weight

            instanceStats["classes"][label.name]["tp"]         += tp
            instanceStats["classes"][label.name]["fn"]         += fn
            instanceStats["classes"][label.name]["tpWeighted"] += tpWeighted
            instanceStats["classes"][label.name]["fnWeighted"] += fnWeighted

            category = label.category
            if category in instanceStats["categories"]:
                catTp = 0
                catTp = np.count_nonzero( np.logical_and( mask , categoryMasks[category] ) )
                catFn = instSize - catTp

                catTpWeighted = float(catTp) * weight
                catFnWeighted = float(catFn) * weight

                instanceStats["categories"][category]["tp"]         += catTp
                instanceStats["categories"][category]["fn"]         += catFn
                instanceStats["categories"][category]["tpWeighted"] += catTpWeighted
                instanceStats["categories"][category]["fnWeighted"] += catFnWeighted

	if args.evalPixelAccuracy:
		notIgnoredLabels = [l for l in args.evalLabels if not id2label[l].ignoreInEval]
		notIgnoredPixels = np.in1d( groundTruthNp , notIgnoredLabels , invert=True ).reshape(groundTruthNp.shape)
		erroneousPixels = np.logical_and( notIgnoredPixels , ( predictionNp != groundTruthNp ) )
		perImageStats[predictionImgFileName] = {}
		perImageStats[predictionImgFileName]["nbNotIgnoredPixels"] = np.count_nonzero(notIgnoredPixels)
		perImageStats[predictionImgFileName]["nbCorrectPixels"]    = np.count_nonzero(erroneousPixels)

	return nbPixels

def run_eval(resultPath):
	global args

	args.predictionPath = resultPath
	predictionImgList = []
	groundTruthImgList = []
	avgScore = 0.0

	groundTruthImgList = glob.glob(args.groundTruthSearch)
	if not groundTruthImgList:
		printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(args.groundTruthSearch))
        # get the corresponding prediction for each ground truth imag
	for gt in groundTruthImgList:
		predictionImgList.append( getPrediction(args, gt) )

	print('load all resources done! Start evaluating ...')
	# evaluate
	#print('list of predictions: ', predictionImgList)
	#print('list of truth: ', groundTruthImgList)
	print('predictions %d truth %d: '%(len(predictionImgList), len(groundTruthImgList)))
	avgScore = evaluateImgLists(predictionImgList, groundTruthImgList, args)

	print('evaluation done!')

	return avgScore


