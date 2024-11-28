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

function [module,orien]=getgradsobel(I,N)
N2=round(N/2);
% Sobel 3x3
fx = [1 0 -1];
fy = [1 3 1]';

Ix=conv2(conv2(double(I),fx,'same'),fy,'same');
Iy=conv2(conv2(double(I),fx','same'),fy','same');
orien=mod(round(N*atan2(Iy,-Ix)/pi),N) + 1; % quantize radian values into N integers
module=sqrt(Ix.*Ix+Iy.*Iy);