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

function varargout = LPRN_Demo(varargin)
% LPRN_DEMO MATLAB code for LPRN_Demo.fig
%      LPRN_DEMO, by itself, creates a new LPRN_DEMO or raises the existing
%      singleton*.
%
%      H = LPRN_DEMO returns the handle to a new LPRN_DEMO or the handle to
%      the existing singleton*.
%
%      LPRN_DEMO('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LPRN_DEMO.M with the given input arguments.
%
%      LPRN_DEMO('Property','Value',...) creates a new LPRN_DEMO or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LPRN_Demo_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LPRN_Demo_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LPRN_Demo

% Last Modified by GUIDE v2.5 15-Aug-2017 08:45:00

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @LPRN_Demo_OpeningFcn, ...
                   'gui_OutputFcn',  @LPRN_Demo_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before LPRN_Demo is made visible.
function LPRN_Demo_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to LPRN_Demo (see VARARGIN)

% Choose default command line output for LPRN_Demo
handles.output = hObject;

axes(handles.axes1);
axis off;
axes(handles.axes2);
axis off;
addpath('SVMCode');
addpath(fullfile('SVMCode','HoGFunctions'));
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes LPRN_Demo wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = LPRN_Demo_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in upload.
function upload_Callback(hObject, eventdata, handles)
% hObject    handle to upload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[FileName,PathName,FilterIndex] = uigetfile('*.*');
handles.file = fullfile(PathName,FileName);
handles.img = rgb2gray(imread(handles.file));
axes(handles.axes1);
imshow(handles.img);
set(handles.reliability_edit,'String','');
set(handles.timeElapsed_edit,'String','');
set(handles.msgCrop_text,'String','');
% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in cropButton.
function cropButton_Callback(hObject, eventdata, handles)
% hObject    handle to cropButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.msgCrop_text,'String','Double click to finish.');
axes(handles.axes1);
h = imrect;
p = round(wait(h));
handles.Crop = handles.img(p(2):p(2)+p(4)-1,p(1):p(1)+p(3)-1);
axes(handles.axes2);
imshow(handles.Crop);
delete(h);
axes(handles.axes1);
drawRectangle(p,'r',2);
set(handles.timeElapsed_edit,'String','');
set(handles.reliability_edit,'String','');
set(handles.msgCrop_text,'String','');
% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in recogButton.
function recogButton_Callback(hObject, eventdata, handles)
% hObject    handle to recogButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Clean Boxes
set(handles.KNN_number_edit,'String','#');
set(handles.KNN_timeElapsed_edit,'String','');
set(handles.number_edit,'String','#');
set(handles.reliability_edit,'String','');
set(handles.timeElapsed_edit,'String','');
% Disable buttons
set(handles.recogButton,'Enable','off');
set(handles.upload,'Enable','off');
set(handles.cropButton,'Enable','off');
% update fig
drawnow

LIST_NTRAIN = [20 40];
kernelFunction = get(handles.kernel_popmenu,'Value');

switch kernelFunction
    case 1 % Linear
        str_kernel = 'linear';
    case 2 % Polynomial 
        str_kernel = 'polynomial';
    case 3 % RBF
        str_kernel = 'rbf';
    case 4 % Optimal Linear
        str_kernel = 'linear';
end
% Fix gradient directions
GRADIENT_DIRECTIONS = 4;
% Fix K in Knn
KNN = 5;

% Verify the existence of Features Matrix from License Plate Dataset
idx_nroSamples = get(handles.popupmenu_NroSamples,'Value');
NTRAIN = LIST_NTRAIN(idx_nroSamples);

procRecogNumber(fullfile('SVMCode','data'), 'tmp',handles.Crop,kernelFunction,NTRAIN,GRADIENT_DIRECTIONS,KNN);

% recover outputs and print on GUI
% SVM
number = load(fullfile('tmp','svm_number.dat'),'-ascii');
reliability = load(fullfile('tmp','svm_reliability.dat'),'-ascii');
tElapsed = load(fullfile('tmp','svm_tElapsed.dat'),'-ascii');
set(handles.number_edit,'String',num2str(number));
set(handles.reliability_edit,'String',num2str(reliability));
set(handles.timeElapsed_edit,'String',num2str(tElapsed));
% KNN
% recover outputs
knn_number = load(fullfile('tmp','knn_number.dat'),'-ascii');
tElapsed = load(fullfile('tmp','knn_tElapsed.dat'),'-ascii');
set(handles.KNN_number_edit,'String',num2str(knn_number));
set(handles.KNN_timeElapsed_edit,'String',num2str(tElapsed));

% clean temporal folder
delete(fullfile('tmp','svm_number.dat'),fullfile('tmp','svm_reliability.dat'),...
    fullfile('tmp','svm_tElapsed.dat'),...
    fullfile('tmp','knn_number.dat'),fullfile('tmp','knn_tElapsed.dat'));
% Enable buttons
set(handles.recogButton,'Enable','on');
set(handles.upload,'Enable','on');
set(handles.cropButton,'Enable','on');
% update fig
drawnow

% Update handles structure
guidata(hObject, handles);


function number_edit_Callback(hObject, eventdata, handles)
% hObject    handle to number_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of number_edit as text
%        str2double(get(hObject,'String')) returns contents of number_edit as a double


% --- Executes during object creation, after setting all properties.
function number_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to number_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function reliability_edit_Callback(hObject, eventdata, handles)
% hObject    handle to reliability_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of reliability_edit as text
%        str2double(get(hObject,'String')) returns contents of reliability_edit as a double


% --- Executes during object creation, after setting all properties.
function reliability_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to reliability_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function timeElapsed_edit_Callback(hObject, eventdata, handles)
% hObject    handle to timeElapsed_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of timeElapsed_edit as text
%        str2double(get(hObject,'String')) returns contents of timeElapsed_edit as a double


% --- Executes during object creation, after setting all properties.
function timeElapsed_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to timeElapsed_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in kernel_popmenu.
function kernel_popmenu_Callback(hObject, eventdata, handles)
% hObject    handle to kernel_popmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns kernel_popmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from kernel_popmenu


% --- Executes during object creation, after setting all properties.
function kernel_popmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kernel_popmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu_NroSamples.
function popupmenu_NroSamples_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_NroSamples (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_NroSamples contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_NroSamples


% --- Executes during object creation, after setting all properties.
function popupmenu_NroSamples_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_NroSamples (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function KNN_number_edit_Callback(hObject, eventdata, handles)
% hObject    handle to KNN_number_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of KNN_number_edit as text
%        str2double(get(hObject,'String')) returns contents of KNN_number_edit as a double


% --- Executes during object creation, after setting all properties.
function KNN_number_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to KNN_number_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function KNN_timeElapsed_edit_Callback(hObject, eventdata, handles)
% hObject    handle to KNN_timeElapsed_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of KNN_timeElapsed_edit as text
%        str2double(get(hObject,'String')) returns contents of KNN_timeElapsed_edit as a double


% --- Executes during object creation, after setting all properties.
function KNN_timeElapsed_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to KNN_timeElapsed_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
