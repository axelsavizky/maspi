// This file is part of the implementation on MATLAB of the Platt pseudo 
// code published on "Sequential Minimal Optimization: A Fast Algorithm
// for Training Support Vector Machine" paper.
// 
// Copyright(c) 2016 Pablo Negri
// pnegri@uade.edu.ar
// 
// This file may be licensed under the terms of of the
// GNU General Public License Version 2 (the ``GPL'').
// 
// Software distributed under the License is distributed
// on an ``AS IS'' basis, WITHOUT WARRANTY OF ANY KIND, either
// express or implied. See the GPL for the specific language
// governing rights and limitations.
// 
// You should have received a copy of the GPL along with this
// program. If not, go to http://www.gnu.org/licenses/gpl.html
// or write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.


#include "math.h"
#include "mex.h"

void getRectangleSingleHoGFeature(double * rect, mxArray * hhData,double * ft_hog, double * h, int N, int u , int v);

void mexFunction(int nlhs, mxArray *plhs[ ],int nrhs, const mxArray *prhs[ ]) {

	// input data
	mxArray * rectData, * hhData, * NData, * index_HogData;
	double * rectValues, * index_HogValues;
	// output data
	double * hog_data_values;

	double * h,* ft_hog;
	int u,v,n_fts,i,j,N,n_elem;

	// get input arguments
	rectData = (mxArray *) prhs[0];
	hhData = (mxArray *) prhs[1];
	NData = (mxArray *) prhs[2];
	index_HogData = (mxArray *) prhs[3];

	rectValues = mxGetPr(rectData);
	index_HogValues = mxGetPr(index_HogData);

	N = (int)(mxGetScalar(NData));

	u = mxGetDimensions(hhData)[0];
	v = mxGetDimensions(hhData)[1];

	n_fts = mxGetDimensions(index_HogData)[0];
	n_elem = mxGetDimensions(index_HogData)[1];

	// allocate output data
	plhs[0] = mxCreateDoubleMatrix(n_fts, N, mxREAL); //mxReal is our data-type
	//Get a pointer to the data space in our newly allocated memory
	hog_data_values = mxGetPr(plhs[0]);
	
	h = (double *) calloc(sizeof(double),N);
	ft_hog = (double *) malloc(sizeof(double)*n_elem);


	for(i=0;i<n_fts;i++)
	{
		for(j=0;j<n_elem;j++)
        {
			ft_hog[j] = index_HogValues[(j*n_fts)+i];
        }

		getRectangleSingleHoGFeature(rectValues,hhData,ft_hog,h,N,u,v);

		for(j=0;j<N;j++)
			hog_data_values[(j*n_fts) +i] = h[j];
	}

	free(h);
	free(ft_hog);

	return;
}


void getRectangleSingleHoGFeature(double * rect, mxArray * hhData,double * ft_hog, double * h, int N, int u , int v)
{
// hh : histogram integrale
// ft_HoG
// ws : window size
	double * hh;
	int i,x,y,dx,dy,x1,x2,x3,x4,type;
	double sc=ft_hog[2];
	double cum_sum = 0;
   
// Feature 
	x=rect[1]-1+(ft_hog[3]-1);
	y=rect[0]-1+(ft_hog[4]-1);
    type = ft_hog[1];
    
    switch(type)
    {
        case 1: // cuadrados
            dx = sc;
            dy = dx;
            break;
        case 2: // rectangulos verticales
            dx = sc / 2;
            dy = sc;
            break;
        case 3: // rectangulos horizontales
            dx = sc;
            dy = sc / 2;
            break;
    }

	hh = mxGetPr(hhData);
	
	x1 = y + x*u;
	x2 = y + (x+dx)*u;
	x3 = (y+dy) + (x+dx)*u;
	x4 = (y+dy) + x*u;

	for(i=0;i<N;i++) 
		h[i] = fabs((hh[u*v*i + x1] + hh[u*v*i + x3]) - 
			(hh[u*v*i + x2] + hh[u*v*i + x4]));
	for(i=0;i<N;i++)
	{
		if(h[i] < 1)
			h[i] = 0;
		cum_sum += h[i];
	}

	if(cum_sum)
	{
		for(i=0;i<N;i++)
			h[i] = h[i] / cum_sum;
	}
	return;
}
