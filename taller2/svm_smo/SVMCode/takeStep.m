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

function output = takeStep(i1,i2)

global E K Targets Alphas C;

e = 0.001;
output = 0;
if  i1 == i2;return;end;

alph1 = Alphas(i1);
y1 = Targets(i1);

alph2 = Alphas(i2);
y2 = Targets(i2);

E1 = E(i1);
E2 = E(i2);

s = y1*y2;

% Compute L and H
if s == 1
    L = max([0 alph1+alph2-C]);
    H = min([C alph2+alph1]);
else
    L = max([0 alph2-alph1]);
    H = min([C C+alph2-alph1]);
end

if L == H;return;end;

k11 = K(i1,i1);
k12 = K(i1,i2);
k22 = K(i2,i2);

eta = k11+k22-2*k12;

if eta > 0
    a2 = alph2 + y2 * (E1-E2) / eta;
    a2 = min([max([a2 L]) H]);
else
    % Compute new value of alpha2 and clipped to the constrains
    if (y2 * (E1-E2)) < 0
        a2 = H;
    elseif (y2 * (E1-E2)) > 0
        a2 = L;
    else
        a2 = alph2;
    end
end
% If there is not change, return false
if abs(a2-alph2) < eps*(a2+alph2+eps)
    return;
end

% Compute new value for alpha1
a1 = alph1 + s*(alph2-a2);

% Update threshold to reflect change in Lagrange multipliers
updateThreshold(i1,i2,a1,a2);
% Update weight vector to reflect change in a1 & a2, if SVM is linear
%%%%%%%%%%%%%
Alphas(i1) = a1;
Alphas(i2) = a2;
% Update eror cache using new Lagrange multipliers
updateErrorList;

output = 1;
        



