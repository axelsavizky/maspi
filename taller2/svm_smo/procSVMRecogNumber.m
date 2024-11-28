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

function procSVMRecogNumber(dataFolder, parametersFolder,img,kernelFunction, ...
    NTRAIN, fromDemo)

if ~exist('fromDemo', 'var')
    fromDemo = false;
end

fts = load(fullfile(dataFolder,'index_HoG.mat'));
str_kernel = [];

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

svm_reliability = 0;
svm_number = -1;

Pat = getFeatureVector(img,svm.GRADIENT_DIRECTIONS,fts.index_HoG);
xt = Pat(:);
vec_outputs = zeros(1,10);
tic
for k=1:10
    % Create Targets vector
    Targets = -1*ones(1,length(svm.y));
    idx = find(svm.y == (k-1));
    Targets(idx) = 1;
    
    cum_out = 0;
    if kernelFunction == 4 % Optimal Linear
        cum_out = svm.w_optlin(k,:) * xt;
    else
        for j=1:length(svm.y)
            cum_out = cum_out + computeKernelFunction(svm.x(:,j),xt,kernelFunction,svm.GAMMA,svm.POLYNOMIAL_DEGREE) * svm.TabAlphas(k,j) * Targets(j);        
        end
    end
    vec_outputs(k) = cum_out - svm.TabB(k);
end
[mx,cl_mx] = max(vec_outputs);
svm_number = cl_mx - 1;
tElapsed = toc;

filenameTimeSVM = fullfile(parametersFolder,'svm_tElapsed.dat');
if (fromDemo)
    fid = fopen(filenameTimeSVM,'wt');
    fprintf(fid, '%.4f seconds', tElapsed);
    fclose(fid);
else    
    save(filenameTimeSVM,'tElapsed','-ascii');
end

% CONFIDENCE COMPUTATION
[sort_outputs,srt_mx] = sort(vec_outputs,'descend');
%%%%% ciarp confidences %%%%
m = mean(sort_outputs);
v = var(sort_outputs);
pdf_outputs = (sort_outputs - m).^2 / v;
pdf_outputs = pdf_outputs / sum(pdf_outputs);
cr = pdf_outputs(1) / (sum(pdf_outputs(2:end)));
cd = pdf_outputs(1);

svm_reliability = cr / svm.CR_TH * cd / svm.CD_TH;

filenameSVMNumber = fullfile(parametersFolder,'svm_number.dat');
filenameSVMReliability = fullfile(parametersFolder,'svm_reliability.dat');
if (fromDemo)
    fid = fopen(filenameSVMNumber,'wt');
    fprintf(fid, '%d', svm_number);
    fclose(fid);
    fid = fopen(filenameSVMReliability,'wt');
    fprintf(fid, '%.3f', svm_reliability);
    fclose(fid);
else    
    save(filenameSVMNumber, 'svm_number','-ascii');
    save(filenameSVMReliability, 'svm_reliability','-ascii');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = computeKernelFunction(x1,x2,KernelFunction,GAMMA,POLYNOMIAL_DEGREE)

switch KernelFunction
    case 1 % LINEAR
        f = (x1'*x2);
    case 2 % POLYNOMIAL
        f = (x1'*x2+1)^POLYNOMIAL_DEGREE;
    case 3 % RBF
        f = exp(-GAMMA*((x1-x2)'*(x1-x2)));
end