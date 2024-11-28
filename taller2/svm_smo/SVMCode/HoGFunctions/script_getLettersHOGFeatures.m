% This file is part of the implementation on MATLAB of the Platt pseudo 
% code published on "Sequential Minimal Optimization: A Fast Algorithm
% for Training Support Vector Machine" paper, corresponding to 
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

folder = fullfile('..','Datasets','BaseOCR_MultiStyle');

dchars = {
    '0';
    '1';
    '2';
    '3';
    '4';
    '5';
    '6';
    '7';
    '8';
    '9'}';


Base = [];
labels = [];
name = {};

load(fullfile('..','data','index_HoG.mat'));

for d=1:length(dchars)

    % get file list
    files = dir(fullfile(folder,dchars{d},'*.bmp'));
    files = [files ; dir(fullfile(folder,dchars{d},'*.tif'))];
    tmp_Base = [];
    tmp_Labels = [];
    tmp_name = {};
    for f=1:length(files)
        filename = fullfile(folder,dchars{d},files(f).name);
        disp(filename);
        Ip = double(imread(filename));
    
        Pat = getFeatureVector(Ip,GRADIENT_DIRECTIONS,index_HoG);     
        tmp_Base = [tmp_Base Pat(:)];      
        tmp_Labels = [tmp_Labels d-1];
        tmp_name{end+1} = filename;
    end
    % verify if there are duplicated samples
    nexamples = length(tmp_Labels);
    for f=1:nexamples
        vector = tmp_Base(:,f);
        error_sample = sum([Base tmp_Base] - repmat(vector,1,nexamples+size(Base,2)));
        x = find(error_sample == 0);
        if length(x) == 1
            Base = [Base tmp_Base(:,f)];
            labels = [labels tmp_Labels(f)];
            name{end+1} = tmp_name{f};
        else
            fprintf('(%s) Duplicated sample: %s - %d\n',...
                dchars{d},files(f).name,length(x));
        end
    end
end

save(fullfile('..','data','HOGFeatures.mat'),'Base','labels','name');
