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

function [CR_TH,CD_TH] = computeOCRCogniteThresholds(Xapp,Lapp,TabAlphas,TabB,KernelFunction)

[nfts,nexamples] = size(Xapp);

Cr_correct = [];
Cd_correct = [];

for i=1:nexamples
    %disp([num2str(i) ':' num2str(nexamples)]);
    
    x1 = Xapp(:,i);
    lbl = Lapp(i);
    
    vec_labels = unique(Lapp);
    vec_outputs = zeros(1,length(vec_labels));
    for k=1:length(vec_labels)
        % create targets vector for class k
        Targets = -1*ones(1,length(Lapp));
        idx = find(Lapp == vec_labels(k));
        Targets(idx) = 1;
        
        cum_out = 0;
        for n=1:length(Lapp)
            x2 = Xapp(:,n);
            cum_out = cum_out + computeKernelFunction(x1,x2,KernelFunction) * TabAlphas(k,n) * Targets(n);
        end
        vec_outputs(k) = cum_out - TabB(k);
    end
    [sort_outputs,srt_mx] = sort(vec_outputs,'descend');

    %%%%% ciarp confidences %%%%
    m = mean(sort_outputs);
    v = var(sort_outputs);
    pdf_outputs = (sort_outputs - m).^2 / v;
    % normalize to obtain a pdf
    pdf_outputs = pdf_outputs / sum(pdf_outputs);
%     
    Cr = pdf_outputs(1) / (sum(pdf_outputs(2:end)));
    Cd = pdf_outputs(1);

    if srt_mx(1) == lbl+1
        Cr_correct = [Cr_correct Cr];
        Cd_correct = [Cd_correct Cd];
    end
end
CR_TH = min(Cr_correct);
CD_TH = min(Cd_correct);



