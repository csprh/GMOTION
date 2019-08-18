# This script demonstrates how to generate a list of classifications 
# (and probabilities) from an extracted grid of lat,lon and dates

#
# Firstly the data is adjusted within the datacube generation xml config file
# e.g. configHABunderDesk.xml

# Then then following scripts are called

# test_genAllH5s.m: A MATLAB script that generates a datacube (H5 format) 
# for a grid of lat and lon values

# test_cubeSequence.m: A MATLAB script that generates quantised
# images that are put into outputDirectory (from the datacubes)

# extract_features: A python script that extracts bottle neck features using
# CNNs defined in the config file xml.  The features are stored in
# imsDir

# testHAB: A python script that uses the model defined in the xml file and
# generates a classification based on the datacubes extracted images
# The classifications and probabilities for all datacubes are stored in 
# text file stored in the date based directory (imsDir)..classesProbs.txt
#
# THE UNIVERSITY OF BRISTOL: HAB PROJECT
# Author Dr Paul Hill March 2019

import sys
import os
import extract_features
import testHAB

from xml.etree import ElementTree as et

import pudb; pu.db

sample_date = 737174;
sample_date_string = str(sample_date)

#mstringApp = '/Applications/MATLAB_R2016a.app/bin/matlab'
xmlName = '/home/cosc/csprh/linux/HABCODE/code/HAB/extractData/configHABunderDesk.xml'

mstringApp = 'matlab'

tree = et.parse(xmlName)
tree.find('.//testDate').text = sample_date_string
imsDir = tree.find('.//testImsDir').text
tree.write(xmlName)

imsDir = os.path.join(imsDir, sample_date_string)
modelD = os.getcwd()

os.chdir('../extractData')
# GENERATE DATACUBES FOR A BUNCH OF LAT, LON POSITIONS IN A GRID
mstring = mstringApp + ' -nosplash -r \"test_genAllH5s; quit;\"'
os.system(mstring)
os.chdir('postProcess')
# GENERATE IMAGES FROM DATA CUBES
# GENERATED LAT AND LONS latLonList.txt TEXT FILE IN imsDir
mstring = mstringApp + ' -nosplash -r \"test_cubeSequence; quit;\"'
os.system(mstring)

os.chdir(modelD)
# EXTRACT BOTTLENECK FEATURES FROM IMAGES
extract_features.main(['cnfgXMLs/NASNet11_lstm0.xml', imsDir])

# GENERATE CLASSIFICATION FROM BOTTLENECK FEATURES AND TRAINED MODEL
# GENERATED CLASSIFICATIONS ENTERED INTO classesProbs.txt TEXT FILE IN imsDir
testHAB.main(['cnfgXMLs/NASNet11_lstm0.xml', imsDir])



