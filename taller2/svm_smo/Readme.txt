IPOL SVM SMO MATLAB DEMO 
------------------------

The demo applies a multi-style license plate number classification using a SVM classifier trained with SMO.
This demo corresponds to the article ''A MATLAB SMO Implementation to Train a SVM Classifier: Application to Multi-Style License Plate Numbers Recognition'' version 1.0 IPOL article.

Copyright (c) 2016 by Pablo Negri
License: GNU General Public License Version 3 (the ``GPL'').


Usage
=====

1 - Browse the disks to a License Plata image using ''Upload Image'' button. There are some examples on ''LPImages'' folder.
2 - The button ''Crop Number'' selects a character by dragging a rectangle with the mouse. Double click the left button to finish the selection. The corresponding sub-image will be extracted an copied on the lower panel.
3 - Classify the extracted image using button ''Number Recognition''. The demo shows the character associated with the classifier which obtained the highest SVM output score. It is also showed the reliability measure, which must be greater than 1 to be a reliable ouput, and the elapsed time in seconds.
