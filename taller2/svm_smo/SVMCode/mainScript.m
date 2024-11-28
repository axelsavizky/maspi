% This file is part of the implementation on MATLAB of the Platt pseudo 
% code published on "Sequential Minimal Optimization: A Fast Algorithm
% for Training Support Vector Machine" paper, corresponding to 
% the "A MATLAB SMO Implementation to Train a SVM Classifier: 
% Application to Multi-Style License Plate Numbers Recognition",
% version 1.0 IPOL article
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

global K E Targets Alphas b;

load(fullfile('SVMCode','data','HOGFeatures.mat'));
addpath('HoGFunctions');
% Creating train data and test data.
% NTRAIN samples randomly selected are used for train and the others for tests.
rand('seed',100*NTRAIN);
x = [];
y = [];

% Split randomly samples for training
for i=0:9
    idx = find(labels == i);
    xp = randperm(length(idx));

    % select equal number of samples from each country
    country_count = zeros(4);
    for j=1:length(xp)        
        save_sample = 0;
        if isArg(name{idx(xp(j))}) && country_count(1) < NTRAIN/4
            save_sample = 1;
            country_count(1) = country_count(1) + 1;
        elseif isEEUU(name{idx(xp(j))}) && country_count(2) < NTRAIN/4
            save_sample = 1;
            country_count(2) = country_count(2) + 1;
        elseif isPak(name{idx(xp(j))}) && country_count(3) < NTRAIN/4
            save_sample = 1;
            country_count(3) = country_count(3) + 1;
        elseif country_count(4) < NTRAIN/4
            save_sample = 1;
            country_count(4) = country_count(4) + 1;
        end
        if save_sample
            x = [x Base(:,idx(xp(j)))];
            y = [y labels(idx(xp(j)))];
        end
    end
end
%%%%%%%%%%%% TRAIN %%%%%%%%%%%
disp('Training...');
[nfts,npts] = size(x);
K = zeros(npts);
% Polinomial based Kernel computation from features vectors.
for i=1:npts
    x1 = x(:,i);
    for j=1:npts
        x2 = x(:,j);
        K(i,j) = computeKernelFunction(x1,x2,kernelFunction);
    end
end
% Train each class against all the others and save the alpha values in a
% matrix.
TabAlphas = zeros(10,length(y));
TabB = zeros(1,10);

for i=1:10
    fprintf('Training character number %s \n',num2str(i-1));
    fprintf('------------------------------\n');
    % create Targets vector
    Targets = -1*ones(1,length(y));
    idx = find(y == (i-1));
    Targets(idx) = 1;
    
    b = 0;
    E = -Targets; % error table
    Alphas = zeros(npts,1);

    % run the support vector algorithm
    computeSMOMultipliers(C);
    % save computed variables on Tables
    TabAlphas(i,:) = Alphas';
    TabB(i) = b;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute weight vector for optimal linear kernel
w_optlin = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(str_kernel,'linear')
    % compute the optimization vector
    w_optlin = zeros(10,nfts);
    for k=1:10
        % Create Targets vector
        Targets = -1*ones(1,length(y));
        idx = find(y == (k-1));
        Targets(idx) = 1;
        for j=1:length(y)
            w_optlin(k,:) = w_optlin(k,:) + x(:,j)' * TabAlphas(k,j) * Targets(j);
        end
    end
end
%%%%RELIABILITY MEASURES%%%%%
[CR_TH,CD_TH] = computeOCRCogniteThresholds(x,y,TabAlphas,TabB,kernelFunction);
%%%%%%%%%%%% save %%%%%%%%%%%
classifier_name = ['svmSMOmultipliers_' num2str(NTRAIN*10) '.mat'];
if ~exist(fullfile('SVMCode','data',str_kernel),'dir')
    mkdir(fullfile('SVMCode','data',str_kernel));
end
save(fullfile('SVMCode','data',str_kernel,classifier_name),...
    'x','y','TabB','TabAlphas','CR_TH','CD_TH','POLYNOMIAL_DEGREE','GAMMA','GRADIENT_DIRECTIONS','NTRAIN','w_optlin');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
