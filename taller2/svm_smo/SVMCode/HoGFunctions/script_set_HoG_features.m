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

function script_set_HoG_features
% Calculate the linear HoG index to (x,y) format:
% The output file has 5 columns and each of them means:
% 1-	feature index number
% 2-	the type of the HoG feature (in this implementation this number 
%       is the same for all the features, but I used to have more than 
%       one when I worked with squares and rectangles regions).
% 3-	The size of the square (in pixels)
% 4-	The x position (assuming the patch has size 32x32 pixels)
% 5-	The y position

u=16;v=12;
index_HoG = [];
acc_index = 1;
% cuadrados
for sc=[4 6 8]
    q_w = v - sc + 1;
    q_h = u - sc + 1;

    % Feature 
    for i=1:q_w, % colonnes
        for j=1:q_h, % lines
            index_HoG = [index_HoG ; acc_index 1 sc i j];  % [fts n_fts sc x y]
            acc_index = acc_index+1;
        end
    end
end
% rectangulos verticales
for sc=[4 6 8]
    q_w = v - sc / 2 + 1;
    q_h = u - sc + 1;

    % Feature 
    for i=1:q_w, % colonnes
        for j=1:q_h, % lines
            index_HoG = [index_HoG ; acc_index 2 sc i j];  % [fts n_fts sc x y]
            acc_index = acc_index+1;
        end
    end
end
% rectangulos horizontales
for sc=[4 6 8]
    q_w = v - sc + 1;
    q_h = u - sc / 2 + 1;

    % Feature 
    for i=1:q_w, % colonnes
        for j=1:q_h, % lines
            index_HoG = [index_HoG ; acc_index 3 sc i j];  % [fts n_fts sc x y]
            acc_index = acc_index+1;
        end
    end
end

save(fullfile('..','data','index_HoG.mat'),'index_HoG');
