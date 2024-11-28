% This file is part of the implementation on MATLAB of the Platt pseudo 
% code published on "Sequential Minimal Optimization: A Fast Algorithm
% for Training Support Vector Machine" pape, corresponding to 
% the "A MATLAB SMO Implementation to Train a SVM Classifier: 
% Application to Multi-Style License Plate Numbers Recognition",
% version 1.0 IPOL article.
% 
% Copyright(c) 2018 Pablo Negri
% pnegri@uba.dc.ar
% 
% This file may be licensed under the terms of of the
% GNU General Public License Version 2 (the ``GPL'').
% 
% Software distributed under the License is distributed
% on an ``AS IS'' basis, WITHOUT WARRANTY OF ANY KIND, either
% express or implied. See the GPL for the specific language
% governing rights and limitations.
% 
% You should have received a copy of the GPL along with this
% program. If not, go to http://www.gnu.org/licenses/gpl.html
% or write to the Free Software Foundation, Inc.,
% 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

function procKNNRecogNumber(dataFolder, parametersFolder,img,kernelFunction, ...
    NTRAIN,KNN, fromDemo)

if ~exist('fromDemo', 'var')
    fromDemo = false;
end

fts = load(fullfile(dataFolder, 'index_HoG.mat'));
str_kernel = [];
d = [];
gamma = [];

switch kernelFunction
    case 1 % Linear
        str_kernel = 'linear';
    case 2 % Polynomial 
        str_kernel = 'polynomial';
    case 3 % RBF
        str_kernel = 'rbf';
    case 4 % Optimal Linear
        str_kernel = 'linear';
end
        
classifier_name = ['svmSMOmultipliers_' num2str(NTRAIN*10) '.mat'];
svm = load(fullfile(dataFolder, str_kernel,classifier_name));

reliability = 0;
number = -1;

Pat = getFeatureVector(img,svm.GRADIENT_DIRECTIONS,fts.index_HoG);
xt = Pat(:);

% Knn classification
tic    
knn_number = applyKNNonHOG(svm.x,svm.y,xt,KNN);
tElapsed = toc;
filenameKNNTime = fullfile(parametersFolder,'knn_tElapsed.dat');
filenameKNNNumber = fullfile(parametersFolder,'knn_number.dat');
if fromDemo
    fid = fopen(filenameKNNNumber,'wt');
    fprintf(fid, '%d', knn_number);
    fclose(fid);
    fid = fopen(filenameKNNTime,'wt');
    fprintf(fid, '%.4f seconds', tElapsed);
    fclose(fid);
    
else
    save(filenameKNNTime,'tElapsed','-ascii');
    save(filenameKNNNumber,'knn_number','-ascii');
end
