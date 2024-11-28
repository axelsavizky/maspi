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

function f = computeKernelFunction(x1,x2,KernelFunction)

global POLYNOMIAL_DEGREE GAMMA
f = 0;
switch KernelFunction
    case 1 % linear
        f = (x1'*x2);
    case 2 % polynomial
        f = (x1'*x2+1)^POLYNOMIAL_DEGREE;
    case 3 % rbf
        f = exp(-GAMMA*((x1-x2)'*(x1-x2)));
    case 4 % optilinear
        f = (x1'*x2);
end