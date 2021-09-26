#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))


// Computes the matrix of cumulated means of a matrix of n rows and m columns
void cummean(int m, int n, double* X, double* mean) {
    double nan = sqrt(-1.0);
    double inf = 1.0/0;
    double minf = log(0.0);
    // N = number of elements of the square with corners (0,0) and (i,j)
    // Nup = number of elements of the square with corners (0,0) and (i-1,j) (upper row)
    // idx and upidx are the indices for (i,j) and (i-1,j) in the unidimensional representation
    int N, Nup, idx, upidx;
    double sum;

    // Fill first row
    *mean = sum = *X;
    for (int j=1; j<n; j++) {
        sum += *(X + j);
        *(mean + j) = sum / (j+1);
    }

    // Remaining rows,
    idx = n;
    upidx = 0;
    for (int i = 1; i < m; i++) {
        sum = 0;
        for (int j=0; j < n; j++) {
            N = (i+1) * (j+1);
            Nup = i * (j+1);
            sum += *(X + idx);
            *(mean + idx) = sum / N + (*(mean + upidx) * Nup) / N;
            ++idx, ++upidx;

            /*if (*(mean + idx) == inf || *(mean + idx) == -inf || *(mean + idx) == nan || *(mean + idx) == -nan) {
                printf("(%d,%d): (%d,%d,%f,%f,%f), ", i, j, N, Nup, sum, *(mean + upidx), *(mean + idx));
                for (int k = 0; k < N; k++)
                    printf("\n%f, ", *(X+k));
                //exit(1);
            }*/
        }
    }
}


void window_mean(int m, int n, int w, double* X, double* Xmean, int getmean, double* XWmean) {
    double upsum, leftsum, cornersum, sum;
    int isup, iinf, jsup, jinf, row, rowsup, rowinf, idx = 0;
    if (getmean) {
        cummean(m, n, X, Xmean);
    }

    for (int i = 0; i < m; i++) {
        isup = min(m-1, i+w);
        iinf = max(0, i-w) - 1;
        row = i * n;
        rowsup = isup * n;
        rowinf = iinf * n;
        for (int j = 0; j < n; j++) {
            /*if (i > w && i < m-w-1 && j > w && j < n-w-1) {
                j = n-w;
            }*/
            idx = row + j;
            upsum = leftsum = cornersum = 0;
            jsup = min(n-1, j+w);
            jinf = max(0, j-w) - 1;
            if (iinf > -1) {
                upsum = *(Xmean + rowinf + jsup) * (iinf+1) * (jsup+1);
            }
            if (jinf > -1) {
                leftsum = *(Xmean + rowsup + jinf) * (isup+1) * (jinf+1);
            }
            if (iinf > -1 && jinf > -1) {
                cornersum = *(Xmean + rowinf + jinf) * (iinf+1) * (jinf+1);
            }
            sum = *(Xmean + rowsup + jsup) * (isup+1) * (jsup+1);
            *(XWmean + idx) = (sum - upsum - leftsum + cornersum) / ((isup - iinf) * (jsup - jinf));
        }
    }

    // Faster for complete windows
/*    for (int i = w+1; i < m-w; i++) {
        isup = i+w;
        iinf = i-w-1;
        row = i * n;
        rowsup = isup * n;
        rowinf = iinf * n;
        for (int j = w+1; j < n-w; j++) {
            idx = row + j;
            jsup = j+w;
            jinf = j-w-1;
            upsum = *(Xmean + rowinf + jsup) * (iinf+1) * (jsup+1);
            leftsum = *(Xmean + rowsup + jinf) * (isup+1) * (jinf+1);
            cornersum = *(Xmean + rowinf + jinf) * (iinf+1) * (jinf+1);
            sum = *(Xmean + rowsup + jsup) * (isup+1) * (jsup+1);
            *(XWmean + idx) = (sum - upsum - leftsum + cornersum) / ((isup - iinf) * (jsup - jinf));
        }
    }*/
}

void window_var(int m, int n, int w, double *X, double *Xmean, double *X2mean, int getmeans, double *Xvar) {
    int idx = 0;
    double* XWmean, *X2Wmean, *X2;
    if (getmeans) {
        X2 = (double*)malloc(m * n * sizeof(double));
        for (int i = 0; i < m*n; i++) {
            *(X2 + i) = *(X+i) * *(X+i);
        }
        cummean(m, n, X, Xmean);
        cummean(m, n, X2, X2mean);
        free(X2);
    }
    XWmean = (double *)malloc(m * n * sizeof(double));
    X2Wmean = (double *)malloc(m * n * sizeof(double));
    window_mean(m, n, w, X, Xmean, 0, XWmean);
    window_mean(m, n, w, X2, X2mean, 0, X2Wmean);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            *(Xvar + idx) = *(X2Wmean + idx) -  *(XWmean + idx) * *(XWmean + idx);
            ++idx;
        }
    }

    free(XWmean);
    free(X2Wmean);
}

void lawmlNC(int m, int n, double *X, double sig0, int maxwsize, double *Xdenoised) {
    /*  Xmean: matrix of cumulated means of X,
        X2mean: matrix of cumulated means of X^2,
        S:
        T:
        iinf, isup, jinf, jsup: Indices of window
    */
    double *Xmean, *X2mean, *S, *T;
    int idx = 0;

    // Allocating memory
    Xmean = (double *)malloc(m * n * sizeof(double));
    X2mean = (double *)malloc(m * n * sizeof(double));
    S = (double *)malloc(m * n * sizeof(double));
    T = (double *)malloc(m * n * sizeof(double));

    // Means of X and X^2
    int exec = 1;
    for (int w = 1; w <= maxwsize; w++) {
        window_var(m, n, w, X, Xmean, X2mean, exec, T);
        exec = 0;
        idx = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (w == 1) {
                    *(S + idx) = max(0, *(T + idx) - sig0);
                }
                else {
                    *(S + idx) = min(max(0, *(T + idx) - sig0), *(S + idx));
                }
                ++idx;
            }
        }
    }

    // X*S/(S+sig0)
    idx = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            *(Xdenoised + idx) = *(X + idx) * *(S + idx) / (*(S + idx) + sig0);
            ++idx;
        }
    }

    free(Xmean);
    free(X2mean);
    free(S);
    free(T);
}

static PyObject *waveletsC_lawmlNC(PyObject *self,PyObject *args)
{
	PyObject *imObjectRuido,*imObjectS,*imObjectT,*imObjectSinRuido;
	double sig0;
	int multipleNeighbour, maxw, row, idx;
	PyArrayObject *imagenRuido, *imagenSinRuido;
	PyArg_ParseTuple(args,"OOOOdi",&imObjectRuido, &imObjectS, &imObjectT, &imObjectSinRuido, &sig0, &multipleNeighbour);
	imagenRuido = (PyArrayObject *)imObjectRuido;
	imagenSinRuido = (PyArrayObject *)imObjectSinRuido;
	int m = PyArray_DIMS(imagenRuido)[0];
	int n = PyArray_DIMS(imagenRuido)[1];
    double* X = (double *)malloc(m*n * sizeof(double));
    double* Xdenoised = (double *)malloc(m * n * sizeof(double));
    maxw = 1;
    if (multipleNeighbour == 1) {
        maxw = 4;
    }

/*    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", *(double*)PyArray_GETPTR2(imagenRuido,i,j));
        }
        printf("\n");
    }*/

    // printf("********** SIZE OF double: %d", sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            *(X + i*n + j) = *(float*)PyArray_GETPTR2(imagenRuido,i,j);

        }
    }

    lawmlNC(m, n, X, sig0, maxw, Xdenoised);

    for(int i=0; i < m; i++){
        for(int j=0; j < n; j++){
			*(float*)PyArray_GETPTR2(imagenSinRuido,i,j) = (float)*(Xdenoised + i*n + j);
			//printf("%f ", *(float*)PyArray_GETPTR2(imagenSinRuido,i,j));
        }
	}
    free(X);
    free(Xdenoised);
    return Py_BuildValue("i",0);
}

static PyMethodDef waveletsC2Methods[] = {
 {"lawmlNC2", waveletsC_lawmlNC, METH_VARARGS, "Extracci�n de ruido de imagen"},
 {NULL, NULL, 0, NULL}
 };

void initwaveletsC2(void){
 (void) Py_InitModule("waveletsC2", waveletsC2Methods);
 }
