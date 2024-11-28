This is the source code and tools corresponding to the "A MATLAB SMO Implementation to Train a SVM Classifier: Application to Multi-Style License Plate Numbers Recognition" version 1.0 IPOL article.

Copyright (c) 2016 by Pablo Negri
License: GNU General Public License Version 3 (the ``GPL'').

This is the source code implementing Sequential Minimal Optimization algorithm employed to train a Support Vector Machine classifiers.
The code is applied on a multi-style license plate number recognition framework.

The MATLAB mainScript.m file trains and tests the framework. 
It compiles and computes the necesary MEX files, and HOG descriptors automatically.
The output of the system is the confusion matrix of the multi-class classifier results applied on the test dataset.
This script is standalone.

The multi-style svm classifier is saved on folder 'data' on the svmSMOmultipliers.mat file.

The distribution provides a dataset of multi-style license plate numbers. It is placed on the root directory of the mainScript.m file. The hierarchy of folder is:

./Dataset/BaseOCR_MultiStyle/

Inside this folder, the characters numbers are separated by their class number:

./Dataset/BaseOCR_MultiStyle/0
./Dataset/BaseOCR_MultiStyle/1
...
./Dataset/BaseOCR_MultiStyle/9

Each of those folder contains image files of each number. The files can be of BMP, JPG, GIF, PNG, TIFF types.


The MATLAB mainScript4Raw.m file trains and tests the framework but using intensity raw pixel values to train the SVM.
Parameters C and gamma of the RBF kernel are choosen to maximize the results using this features.



http://www.ipol.im
Image Processing On Line