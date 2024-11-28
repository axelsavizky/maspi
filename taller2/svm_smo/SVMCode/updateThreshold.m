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

function updateThreshold(i1,i2,a1,a2)

global K Targets Alphas E b;

alph1 = Alphas(i1);
y1 = Targets(i1);

alph2 = Alphas(i2);
y2 = Targets(i2);

E1 = E(i1);
E2 = E(i2);

b1 = E1 + y1 * (a1 - alph1) * K(i1,i1) + y2 * (a2 - alph2) * K(i1,i2) + b;
b2 = E2 + y1 * (a1 - alph1) * K(i1,i2) + y2 * (a2 - alph2) * K(i2,i2) + b;

if b1 == b2
    b = b1;
else
    b = mean([b1 b2]);
end