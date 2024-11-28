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

function procRecogNumber(dataFolder, parametersFolder, img, kernelFunction, ...
                         NTRAIN, GRADIENT_DIRECTIONS, KNN, fromDemo)

%  dataFolder should point to SVMCode/data if not run from the IPOL demo system.

global POLYNOMIAL_DEGREE GAMMA

if ~exist('fromDemo', 'var')
    fromDemo = false;
else
    % From the IPOL demo, parameters are in string format, so convert them
    % first.
    if ischar(fromDemo) fromDemo = str2num(fromDemo); end
    if ischar(NTRAIN) NTRAIN = str2num(NTRAIN); end
    if ischar(kernelFunction) kernelFunction = str2num(kernelFunction); end
    if ischar(GRADIENT_DIRECTIONS) GRADIENT_DIRECTIONS = str2num(GRADIENT_DIRECTIONS); end
    if ischar(KNN) KNN = str2num(KNN); end
    if ischar(img) img = rgb2gray(imread(img)); end
end

% Verify MEX compilation
mexfile = ['getHogData_c.' mexext];
if (fromDemo && ~exist(mexfile, 'file'))
    fprintf('HoG descriptor library not found.\n');
    exit(0);
end
if ~fromDemo && ~exist(fullfile('SVMCode','HoGFunctions',mexfile),'file') 
    run(fullfile('SVMCode','HoGFunctions','make.m'));
end

% Verify the existence of HOG Features descriptors
if ~exist(dataFolder,'dir')
    mkdir(dataFolder);
end 
if ~exist(fullfile(dataFolder,'index_HoG.mat'),'file')
    if (fromDemo)
        fprintf('index_HoG.mat not found. Please run script_set_HoG_features.m\n');
        exit(0);
    else
        run(fullfile('SVMCode','HoGFunctions','script_set_HoG_features.m'));
    end
end

% Verify the existence of Features Matrix from License Plate Dataset
if ~exist(fullfile(dataFolder,'HOGFeatures.mat'),'file')
    if (fromDemo)
        fprintf('HoGFeatures.mat not found. Please run script_getLettersHOGFeatures.m\n');
        exit(0);
    else
        run(fullfile('SVMCode','HoGFunctions','script_getLettersHOGFeatures.m'));
    end
end

C = 0.1;
switch kernelFunction
    case 1 % Linear
        str_kernel = 'linear';
    case 2 % Polynomial 
        str_kernel = 'polynomial';
    case 3 % RBF
        str_kernel = 'rbf';
        C = 1;
    case 4 % Optimal Linear
        str_kernel = 'linear';
end
classifier_name = ['svmSMOmultipliers_' num2str(NTRAIN*10) '.mat'];
% Verify if the classifier was already trained
if ~exist(fullfile(dataFolder,str_kernel,classifier_name),'file')
    if (fromDemo)
        fprintf('%s not found. Please train classifiers.\n', classifier_name);
        exit(0);
    else
        % Initialize constants
        % For POLYNOMIAL
        POLYNOMIAL_DEGREE   = 2;
        % For RBF
        GAMMA               = 0.01;
        addpath(genpath('SVMCode'));
        mainScript
    end
end

% Now run both classifications

% SVM Classifcation
if ~exist(parametersFolder,'dir')
    mkdir(parametersFolder);
end
% launch scrip
procSVMRecogNumber(dataFolder, parametersFolder,img,kernelFunction,NTRAIN, fromDemo);

% KNN Classifcation
% launch scrip
procKNNRecogNumber(dataFolder, parametersFolder,img,kernelFunction,NTRAIN,KNN, fromDemo)

