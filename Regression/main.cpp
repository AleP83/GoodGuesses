#include "iostream"
#include <gsl/gsl_multifit.h>
#include <random> // for std::random_device and std::mt19937
#include <malloc/malloc.h>
#include <cassert>
#define MAINPATH "/Users/alessandroperi/GoogleDrive/C/Projects/Esperimenti/Regression/"
//#include <cstdio>
//#include <stdio.h>
#define C(i) (gsl_vector_get(c,(i)))
#define COV(i,j) (gsl_matrix_get(cov,(i),(j)))

int REGRESSORS = 5;

typedef struct{
    
    double *******  V;
    double *******  V_imp;
    double *******  gk;
    double *******  gBR;
    double *******  gB;
    double *******  gD;
    double *******  gn;
    int    *******  Index_k;
    int    *******  Index_BR;
    
    
    double *******  V_X;
    //double *******  V_Xbounded;
    double *******  Assetstate;
    double *******  Wealthstate;
    double *******  y;
    double *******  NetInvestment;
    double *******  GrossInvestment;
    
    
    // Firms Decisions
    int    *******  gphiX;
    int    *******  gphiXgphiProduce;
    int    *******  gphiV_Xgreater0;
    int    *******  gphiD;
    int    *******  gphiCh11;
    double *******  Vn;
    //double *******  geffort;
    //double *******  prob_restr;
    double *******  Vc;
    double *******  Vn_imp;
    double *******  Vc_imp;
    /* THRESHOLD
     double ******   ChiThrCh11;
     double ******   ChiThrCh7 ;
     double ******   XThrCh11  ;
     double ******   XThrCh7   ;
     double ******   XThrCh11_imp;
     double ******   XThrCh7_imp ;
     
     
     double ****** X_Vc  ;
     double ****** X_Vn  ;
     double ****** X_VcVn;
     
     double ****** X_Vc_imp  ;
     double ****** X_Vn_imp  ;
     double ****** X_VcVn_imp;
     */
    
    double ******* gBRc;
    double ******* gBc ;
    double ******* gDc ;
    int    ******* Index_BRc;
    
    double ******* gBRn     ;
    double ******* gBn      ;
    double ******* gDn      ;
    int    ******* Index_BRn;
    //int    ******* Index_kn ;
    
    // Labour
    double ******* wage;
    // Recovery Rates - Bargaining
    double ******* Lk;
    double ******* Lk7;
    double ******* Lk11;
    //double ******* alpha_sched;
    double ******* alphaBR;
    double ******* NashBP;
    double ******* NashBP_imp;
    double ******* Chi_restr;
    //double ******* Sf;
    //double ******* Sb;
    
    double *******  V_small;
    int    *******  gphiD_small;
    
    double *** Assetstate_reorg;
    double *** Bank_outsideoption;
    
    // Other
    // int    ****** phiNB_strictgreater0;
    // int    ****** phiNB_Sfstrictgreater0;
    // int    ****** phiNB_Sbstrictgreater0;
    // int    *****  index_debt_Sbstrictgreater0;
    
    // Pricing
    double ***** ri;
    double ***** q;
    double ***** Bp;
    double ***** End_Prob_D;
    double ***** End_Prob_Ch11;
    double ***** End_Prob_Ch7;
    double ***** EPayoff_D;
    double ***** EPayoff_D_Ch11;
    double ***** EPayoff_D_Ch7;
    double ***** EV1;
    
    double ***** Endogenous_Component_Omega_imp;
    double ***** Endogenous_Component_Omega_guess;
    double ***** End_Prob_ND;
    
    
    // Measure
    double measure_mu;     // Measure of incumbents
    double measure_exit;
    double measure_Ch7;
    double measure_Ch11;
    double measure_defexit;// Measure of Default Exit
    
    // Other
    double Ld;
    double error_cumulated_gamma;
    
    // Other
    double max_gk;
    double max_gBR;
    double min_gBR;
    
    
} Incumbent_type;

int NKGRID   = 30;
int NBRGRID  = 30;
int nBRgrid_strictly_pos = 11;
int nBRgrid_neg0 = 19;
int NCHIGRID = 2;
int NXGRID   = 4;
int NCRGRID  = 1;
int NFGRID   = 1;
int NTHETA_AGRID = 1;

double ******* allocate7ddouble(size_t dim_1,size_t dim_2,size_t dim_3,size_t dim_4,size_t dim_5,size_t dim_6,size_t dim_7)
{
    double******* array;
    int i_1,i_2,i_3,i_4,i_5,i_6;
    
    // Check valid dimension: (avoid allocation of size 0 bytes)
    assert(dim_1>0);
    assert(dim_2>0);
    assert(dim_3>0);
    assert(dim_4>0);
    assert(dim_5>0);
    assert(dim_6>0);
    assert(dim_7>0);
    
    // DIMENSION PER DIMENSION
    
    // ----------------------------------------------------------------------------------------------
    array = NULL;
    array= (double*******)malloc(dim_1 * sizeof(double******));
    
    if(array != NULL)
    {
        // ----------------------------------------------------------------------------------------------
        for(i_1=0;i_1<dim_1;i_1++)
        {
            array[i_1] = NULL;
            array[i_1]= (double******)malloc(dim_2 * sizeof(double*****));
            
            
            if(array[i_1]!= NULL)
            {
                
                // ----------------------------------------------------------------------------------------------
                for(i_2=0;i_2<dim_2;i_2++)
                {
                    array[i_1][i_2] = NULL;
                    array[i_1][i_2] = (double*****) malloc(dim_3 * sizeof(double****));
                    if(array[i_1][i_2]!= NULL)
                    {
                        
                        // ----------------------------------------------------------------------------------------------
                        for(i_3=0;i_3<dim_3;i_3++)
                        {
                            array[i_1][i_2][i_3] = NULL;
                            array[i_1][i_2][i_3]= (double****)malloc(dim_4 * sizeof(double***));
                            
                            if(array[i_1][i_2][i_3]!= NULL)
                            {
                                
                                // ----------------------------------------------------------------------------------------------
                                for(i_4=0;i_4<dim_4;i_4++)
                                {
                                    array[i_1][i_2][i_3][i_4] = NULL;
                                    array[i_1][i_2][i_3][i_4]= (double***)malloc(dim_5 * sizeof(double**));
                                    
                                    
                                    if(array[i_1][i_2][i_3][i_4]!= NULL)
                                    {
                                        // ----------------------------------------------------------------------------------------------
                                        for(i_5=0;i_5<dim_5;i_5++)
                                        {
                                            array[i_1][i_2][i_3][i_4][i_5] = NULL;
                                            array[i_1][i_2][i_3][i_4][i_5]= (double**)malloc(dim_6 * sizeof(double*));
                                            
                                            
                                            if(array[i_1][i_2][i_3][i_4][i_5]!= NULL)
                                            {
                                                
                                                // ----------------------------------------------------------------------------------------------
                                                for(i_6=0;i_6<dim_6;i_6++)
                                                {
                                                    array[i_1][i_2][i_3][i_4][i_5][i_6] = NULL;
                                                    array[i_1][i_2][i_3][i_4][i_5][i_6]= (double*)malloc(dim_7 * sizeof(double));
                                                    
                                                    if(array[i_1][i_2][i_3][i_4][i_5][i_6] == NULL)
                                                    {
                                                        printf("Memory not assigned");
                                                        exit(-1);
                                                    } // End if memory not assigned
                                                    
                                                } //i_6
                                                
                                            }
                                            else // if memory not assigned
                                            {
                                                printf("Memory not assigned");
                                                exit(-1);
                                            } // End if memory not assigned
                                            
                                        } // i_5
                                        
                                        
                                    }
                                    else // if memory not assigned
                                    {
                                        printf("Memory not assigned");
                                        exit(-1);
                                    } // End if memory not assigned
                                    
                                    
                                    
                                } // i_4
                                
                            }
                            else // if memory not assigned
                            {
                                printf("Memory not assigned");
                                exit(-1);
                            } // End if memory not assigned
                            
                        } // i_3
                        
                        
                    }
                    else // if memory not assigned
                    {
                        printf("Memory not assigned");
                        exit(-1);
                    } // End if memory not assigned
                    
                    
                } // i_2
                
            }
            else // if memory not assigned
            {
                printf("Memory not assigned");
                exit(-1);
            } // End if memory not assigned
            
        } // i_1
        
    }
    else // if memory not assigned
    {
        printf("Memory not assigned");
        exit(-1);
    }
    
    return array;
}
void deallocate7ddouble(double ******** array,size_t dim_1,size_t dim_2,size_t dim_3,size_t dim_4,size_t dim_5,size_t dim_6)
{
    int i_1,i_2,i_3,i_4,i_5,i_6;
    
    for(i_1=0;i_1<dim_1;i_1++)
    {
        if((*array)[i_1]!=NULL)
        {
            for(i_2=0;i_2<dim_2;i_2++)
            {
                
                if((*array)[i_1][i_2]!=NULL)
                {
                    for(i_3=0;i_3<dim_3;i_3++)
                    {
                        if((*array)[i_1][i_2][i_3]!=NULL)
                        {
                            for(i_4=0;i_4<dim_4;i_4++)
                            {
                                
                                if((*array)[i_1][i_2][i_3][i_4]!=NULL)
                                {
                                    for(i_5=0;i_5<dim_5;i_5++)
                                    {
                                        if((*array)[i_1][i_2][i_3][i_4][i_5]!=NULL)
                                        {
                                            for(i_6=0;i_6<dim_6;i_6++)
                                                if((*array)[i_1][i_2][i_3][i_4][i_5][i_6]!=NULL)
                                                {
                                                    free((*array)[i_1][i_2][i_3][i_4][i_5][i_6]);
                                                    (*array)[i_1][i_2][i_3][i_4][i_5][i_6]=NULL;
                                                    
                                                }
                                            
                                            
                                            free((*array)[i_1][i_2][i_3][i_4][i_5]);
                                            (*array)[i_1][i_2][i_3][i_4][i_5]=NULL;
                                            
                                        }
                                    }
                                    
                                    free((*array)[i_1][i_2][i_3][i_4]);
                                    (*array)[i_1][i_2][i_3][i_4]=NULL;
                                }
                            }
                            
                            free((*array)[i_1][i_2][i_3]);
                            (*array)[i_1][i_2][i_3]=NULL;
                            
                        }
                    }
                    free((*array)[i_1][i_2]);
                    (*array)[i_1][i_2]=NULL;
                    
                }
            }
            free((*array)[i_1]);
            (*array)[i_1]=NULL;
            
        }
    }
    free((*array));
    (*array) = NULL;
}

int main()
{
    int i_theta_A = 0;
    int i_cr=0;
    int i_f = 0;
    
    int i_Chi,i_k,i_BR,i_x;
    
    Incumbent_type   Ri_;
    Ri_.V               = allocate7ddouble(NKGRID,NBRGRID,NXGRID,NCHIGRID,NTHETA_AGRID,NCRGRID,NFGRID);
    Ri_.Vc = allocate7ddouble(NKGRID,nBRgrid_strictly_pos,NXGRID,NCHIGRID,NTHETA_AGRID,NCRGRID,NFGRID);
    std::random_device rd; // Use a hardware entropy source if available, otherwise use PRNG
    //std::mt19937_64 mersenne(rd()); // initialize our mersenne twister with a random seed 64-bit unsigned integers
    std::mt19937_64 mersenne(100); // initialize our mersenne twister with THE SEED 100
    // PRNG NOT A GOOD RANDOM NUMBER GENERATORL: Generating PRNGs that produce uniform results is difficult, and it’s one of the main reasons the PRNG we wrote at the top of this lesson isn’t a very good PRNG.
    std::uniform_real_distribution<double> dis(0.0, 1.0); // Needed to Normalizet the number.
    //See here: http://stackoverflow.com/questions/22923551/generating-number-0-1-using-mersenne-twister-c
    
    //double * kgrid  = new double[NKGRID]
    double kgrid[] = {
        .100000, 15.062069, 30.024138, 44.986207, 59.948276, 74.910345, 89.872414, 104.834483, 119.796552, 134.758621, 149.720690, 164.682759, 179.644828, 194.606897, 209.568966, 224.531034, 239.493103, 254.455172, 269.417241, 284.379310, 299.341379, 314.303448, 329.265517, 344.227586, 359.189655, 374.151724, 389.113793, 404.075862, 419.037931, 434.000000
    };
    /*
    for(i_BR=0;i_BR<NKGRID;i_BR++)
        std::cout << kgrid[i_BR] <<  " , " << i_BR  << "\n";
    */
    
    //double * BRgrid = new double[NBRGRID]
    double BRgrid[] = {
        -894.000000, -844.333333, -794.666667, -745.000000, -695.333333, -645.666667, -596.000000, -546.333333, -496.666667, -447.000000, -397.333333, -347.666667, -298.000000, -248.333333, -198.666667,-149.000000, -99.333333, -49.666667, 0.000000, 45.727273, 91.454545, 137.181818, 182.909091, 228.636364, 274.363636, 320.090909, 365.818182, 411.545455, 457.272727, 503.000000
    };
    /*
    for(i_BR=0;i_BR<NBRGRID;i_BR++)
        std::cout << BRgrid[i_BR] <<  " , " << i_BR  << "\n";
    */
    
    //double * xgrid  = new double[NXGRID]{
    double xgrid[] = {
         0.038774, 0.338465, 2.954512, 25.790340
    };
    
    //double * Chigrid  = new double[NCHIGRID] { 0. , 9.};
    double Chigrid[] = { 0. , 9.};
    
    for(i_k=0;i_k<NKGRID;i_k++)
        for(i_BR=0;i_BR<NBRGRID;i_BR++)
            for(i_x=0;i_x<NXGRID;i_x++)
                for(i_Chi=0;i_Chi<NCHIGRID;i_Chi++)
                {
                    Ri_.V[i_k][i_BR][i_x][i_Chi][i_theta_A][i_cr][i_f] = (double) dis(mersenne);
                    
                    if(i_BR>=nBRgrid_neg0)
                    {
                        int i_BR_debt = i_BR-nBRgrid_neg0;
                        Ri_.Vc[i_k][i_BR_debt][i_x][i_Chi][i_theta_A][i_cr][i_f] = Ri_.V[i_k][i_BR][i_x][i_Chi][i_theta_A][i_cr][i_f];
                    }
                    //std::cout << Ri_.V[i_k][i_BR][i_x][i_Chi][i_theta_A][i_cr][i_f];
                }
    
    
    //https://www.gnu.org/software/gsl/manual/html_node/Fitting-Examples-for-multi_002dparameter-linear-regression.html
    
    gsl_matrix *X, *cov;
    gsl_vector *y, *w, *c;
    
    int ndim = NKGRID * NBRGRID * NXGRID * NCHIGRID;
    
    X = gsl_matrix_alloc (ndim, REGRESSORS);
    y = gsl_vector_alloc (ndim);
    w = gsl_vector_alloc (ndim);
    c = gsl_vector_alloc (REGRESSORS);
    cov = gsl_matrix_alloc (REGRESSORS,REGRESSORS);

    
    // Uploading the Matrixes
    for(i_k=0;i_k<NKGRID;i_k++)
        for(i_BR=0;i_BR<NBRGRID;i_BR++)
            for(i_x=0;i_x<NXGRID;i_x++)
                for(i_Chi=0;i_Chi<NCHIGRID;i_Chi++)
                {
                    // Value Function
                    int indx =
                    i_k         * (NBRGRID*NXGRID*NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID)
                    + i_BR      *         (NXGRID*NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID)
                    + i_x       *                (NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID)
                    + i_Chi     *                         (NTHETA_AGRID*NCRGRID*NFGRID)
                    + i_theta_A *                                      (NCRGRID*NFGRID)
                    + i_cr      *                                              (NFGRID)
                    + i_f;
                    
                    if(i_BR>=nBRgrid_neg0)
                    {
                        int i_BR_debt = i_BR-nBRgrid_neg0;
                        gsl_vector_set (y, indx, Ri_.Vc[i_k][i_BR_debt][i_x][i_Chi][i_theta_A][i_cr][i_f]);
                    }
                    else
                    {
                        gsl_vector_set (y, indx, Ri_.V[i_k][i_BR][i_x][i_Chi][i_theta_A][i_cr][i_f]);
                    }
                    
                    gsl_vector_set (w, indx, 1.0);
                    gsl_matrix_set (X, indx, 0, 1.0);
                    // Kgrid
                    int base,remainder;
                    base = indx / (NBRGRID*NXGRID*NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID);
                    gsl_matrix_set (X, indx, 1, kgrid[base]);
                    remainder = indx % (NBRGRID*NXGRID*NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID);
                    
                    base = remainder / (NXGRID*NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID);
                    gsl_matrix_set (X, indx, 2, BRgrid[base]);
                    remainder = remainder % (NXGRID*NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID);
                    
                    base = remainder / (NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID);
                    gsl_matrix_set (X, indx, 3, xgrid[base]);
                    remainder = remainder % (NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID);
                    
                    base = remainder / (NTHETA_AGRID*NCRGRID*NFGRID);
                    gsl_matrix_set (X, indx, 4, Chigrid[base]);
                    
                }
                    
    
    
    
    
    
    /*
    for (int i = 0; i < ndim; i++)
    {
        
        int indx =
        i_k         * (NBRGRID*NXGRID*NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID)
        + i_BR      *         (NXGRID*NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID)
        + i_x       *                (NCHIGRID*NTHETA_AGRID*NCRGRID*NFGRID)
        + i_Chi     *                         (NTHETA_AGRID*NCRGRID*NFGRID)
        + i_theta_A *                                      (NCRGRID*NFGRID)
        + i_cr      *                                              (NFGRID)
        + i_f;
        
        
        
        gsl_matrix_set (X, i, 0, 1.0);
        gsl_matrix_set (X, i, 1, xi);
        gsl_matrix_set (X, i, 2, xi*xi);
    
        gsl_vector_set (y, i, yi);
        gsl_vector_set (w, i, 1.0);
    }
    */
    
    FILE *fp;
    char filename     [150] ={'\0'};
    strcpy(filename,MAINPATH);
    strcat(filename,"Matrix.txt");
    fp = fopen(filename,"w");
    
    for (int i = 0; i < ndim; i++)
    {
        printf("%d: %6.5lf %6.5lf %6.5lf %6.5lf %6.5lf %6.5lf\n"
               ,i
               ,gsl_vector_get (y, i)
               ,gsl_matrix_get (X, i, 0)
               ,gsl_matrix_get (X, i, 1)
               ,gsl_matrix_get (X, i, 2)
               ,gsl_matrix_get (X, i, 3)
               ,gsl_matrix_get (X, i, 4)
               );
        
        fprintf(fp,"%32.30lf %32.30lf %32.30lf %32.30lf %32.30lf %32.30lf\n"
               ,gsl_vector_get (y, i)
               ,gsl_matrix_get (X, i, 0)
               ,gsl_matrix_get (X, i, 1)
               ,gsl_matrix_get (X, i, 2)
               ,gsl_matrix_get (X, i, 3)
               ,gsl_matrix_get (X, i, 4)
               );
    }
    
    fclose(fp);
    
    // Regression
    double xi, yi, ei, chisq;
    gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (ndim, REGRESSORS);
    gsl_multifit_wlinear (X, w, y, c, cov,&chisq, work);
    gsl_multifit_linear_free (work);
    printf ("# best fit: Y = %g + %g k + %g BR + %g x + %g Chi \n",
            C(0), C(1), C(2),C(3), C(4));
    /*
    printf ("# covariance matrix:\n");
    printf ("[ %+.5e, %+.5e, %+.5e  \n",
            COV(0,0), COV(0,1), COV(0,2));
    printf ("  %+.5e, %+.5e, %+.5e  \n",
            COV(1,0), COV(1,1), COV(1,2));
    printf ("  %+.5e, %+.5e, %+.5e ]\n",
            COV(2,0), COV(2,1), COV(2,2));
    printf ("# chisq = %g\n", chisq);
    */
    deallocate7ddouble(&(Ri_.V)              ,NKGRID,NBRGRID,NXGRID,NCHIGRID,NTHETA_AGRID,NCRGRID);
    deallocate7ddouble(&(Ri_.Vc)              ,NKGRID,nBRgrid_strictly_pos,NXGRID,NCHIGRID,NTHETA_AGRID,NCRGRID);

    /*
    delete [] BRgrid;
    delete [] kgrid;
    delete [] xgrid;
    delete [] Chigrid;
    */
    gsl_matrix_free (X);
    gsl_vector_free (y);
    gsl_vector_free (w);
    gsl_vector_free (c);
    gsl_matrix_free (cov);
    
    std::cout << "Done!";
    return 0;
}