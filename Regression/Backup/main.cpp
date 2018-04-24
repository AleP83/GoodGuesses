#include "iostream"
#include <gsl/gsl_multifit.h>
//#include <cstdio>
//#include <stdio.h>
#define C(i) (gsl_vector_get(c,(i)))
#define COV(i,j) (gsl_matrix_get(cov,(i),(j)))

int main()
{
    
    //https://www.gnu.org/software/gsl/manual/html_node/Fitting-Examples-for-multi_002dparameter-linear-regression.html
    
    std::cout << "Hello world!\n";
    
    gsl_matrix *X, *cov;
    gsl_vector *y, *w, *c;
    
    int nkgrid  = 5;
    int nBRgrid = 1;
    int nxgrid  = 1;
    int nChigrid= 1;
    
    int ndim = nkgrid * nBRgrid * nxgrid * nChigrid;
    
    
    
    X = gsl_matrix_alloc (ndim, 3);
    y = gsl_vector_alloc (ndim);
    w = gsl_vector_alloc (ndim);
    
    /*
    // Uploading the Matrixes
    for (int i = 0; i < ndim; i++)
    {
        gsl_matrix_set (X, i, 0, 1.0);
        gsl_matrix_set (X, i, 1, xi);
        gsl_matrix_set (X, i, 2, xi*xi);
    
        gsl_vector_set (y, i, yi);
        gsl_vector_set (w, i, 1.0);
    }
    
    
    // Regression
    double xi, yi, ei, chisq;
    gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (ndim, 3);
    gsl_multifit_wlinear (X, w, y, c, cov,&chisq, work);
    gsl_multifit_linear_free (work);
    
    {
        printf ("# best fit: Y = %g + %g X + %g X^2\n",
                C(0), C(1), C(2));
        
        printf ("# covariance matrix:\n");
        printf ("[ %+.5e, %+.5e, %+.5e  \n",
                COV(0,0), COV(0,1), COV(0,2));
        printf ("  %+.5e, %+.5e, %+.5e  \n",
                COV(1,0), COV(1,1), COV(1,2));
        printf ("  %+.5e, %+.5e, %+.5e ]\n",
                COV(2,0), COV(2,1), COV(2,2));
        printf ("# chisq = %g\n", chisq);
    }
    
    */
    
    
    
    gsl_matrix_free (X);
    gsl_vector_free (y);
    gsl_vector_free (w);
    gsl_vector_free (c);
    gsl_matrix_free (cov);
    
    return 0;
}