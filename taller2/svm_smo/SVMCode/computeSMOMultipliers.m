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

function computeSMOMultipliers(C)

global Alphas E;

nexamples = length(Alphas);

numChanged = 0;
examineAll = 1;
iteration = 1;
while numChanged > 0 || examineAll
    
    fprintf('Iteration %d\n',iteration);
    
    numChanged = 0;
    if examineAll
        for I=1:nexamples
            numChanged = numChanged + examineExample(I,C);
        end
    else
        for I=1:nexamples
            if Alphas(I) > 0 && Alphas(I) < C
                numChanged = numChanged + examineExample(I,C);
            end
        end
    end
    
    qZAlphas = length(find(Alphas == 0));
    qCAlphas = length(find(Alphas == C));
    
    fprintf('Max. Step Size................: \t %3.3f \n',max(E)-min(E));
    fprintf('Nro. Support Vectors..........: \t %d \n',length(Alphas) - qZAlphas - qCAlphas);
    fprintf('Nro. Zero Alphas..............: \t %d \n', qZAlphas);
    fprintf('Nro. Bounded Alphas (alpha=C).: \t %d \n', qCAlphas);
    fprintf('numChanged....................: \t %d \n',numChanged);
    fprintf('examineAll....................: \t %d \n',examineAll);
    fprintf('+--------------------------------------------------------+\n');
    if examineAll == 1
        examineAll = 0;
    elseif numChanged == 0
        examineAll = 1;
    end
    
    iteration = iteration + 1;
    % whatchdog
    if iteration > 1000
        break;
    end
       
end
