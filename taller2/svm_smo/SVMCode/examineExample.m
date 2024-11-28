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

function output = examineExample(i2,C)

global E Targets Alphas;

output = 0;
y2 = Targets(i2);
alph2 = Alphas(i2);

E2 = E(i2);

r2 = E2*y2;

if (r2 < -0.001 && alph2 < C) || (r2 > 0.001 && alph2 > 0)
    indx_nz_nC = find(Alphas > 0 & Alphas < C);
    % Use Second Heuristic to choose the sample in order to maxmize the
    % error step.
    if length(indx_nz_nC) > 1
        i1 = secondChoiceHeuristic(i2);
        if takeStep(i1,i2)
            output = 1;
            return;
        end
    end    
    % loop over all non-zero and non-C alpha, starting at a random point
    rand_indx = randperm(length(indx_nz_nC));
    for j=1:length(indx_nz_nC)
        i1 = indx_nz_nC(rand_indx(j));
        if takeStep(i1,i2)
            output = 1;
            return;
        end
    end
    % loop over all posible i1, starting at a random point
    rand_indx = randperm(length(Targets));
    for j=1:length(Targets)
        i1 = rand_indx(j);
        if takeStep(i1,i2)
            output = 1;
            return;
        end
    end
end

  

    