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

function fts = getFeatureVector(img_letter,N,index_HoG)
DY = 16;
DX = 12;

% resize number image to pattern size
img_letter = imresize(double(img_letter),[DY DX]);
% extend the size of the pattern to avoid gradient high values on image boundaries 
img_bg = double(imresize(img_letter,[DY+2 DX+2]));

img_bg(2:end-1,2:end-1) = img_letter;

[m,o]=getgradsobel(img_bg,N);
% clean bords
M = m(2:end-1,2:end-1);
O = o(2:end-1,2:end-1);

fts = getHogFeatures(M,O,N,index_HoG);
