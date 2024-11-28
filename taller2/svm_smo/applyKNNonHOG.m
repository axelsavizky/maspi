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

function outLabel = applyKNNonHOG(x,y,xtest,KNN)

% get dataset length
nbase = length(y);
% labels vector
vectLab = unique(y);
% compute all distances
dst = getHistogramCorrelationDistance(x,repmat(xtest,1,nbase));
% sort distances
[s,idxmin] = sort(dst);

% now, compute the knn result
cumclasses = zeros(1,10);
vecSort = y(idxmin(1:KNN));
u = unique(vecSort);
for j=1:length(u)
    cumclasses(u(j)+1) = length(find(vecSort == u(j)));
end
[mxVotes,idx] = max(cumclasses);
outLabel = vectLab(idx);

