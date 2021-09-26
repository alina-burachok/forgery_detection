#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>

void submatrix(PyArrayObject* matriz, int N, int i, int j,double*** subMatriz, int *lenghtSubmatrizX, int *lenghtSubmatrizY) 
 {
	/*OJO liberar memoria reservada despues de usar*/
	/*Crea una submatriz cuadrada de tama�o N de la matriz "matriz" centrada en i,j devolviendola en "subMatriz". Puede ser que la submatriz
	que se devuelva no sea cuadrada, por ejemplo en el caso de las esquinas y laterales. Se devuelve las longitudes de las submatriz generada.
	Se reseva memoria din�micamente para la submatriz asi que HAY QUE LIBERARLA.
	*/
	N=N/2;
	int lenghtMatrizX = PyArray_DIMS(matriz)[0];
	int lenghtMatrizY = PyArray_DIMS(matriz)[1];
	int i0= fmax(0,i-N);
	int j0= fmax(0,j-N);
	int i1= fmin(lenghtMatrizX-1,i+N);
	int j1= fmin(lenghtMatrizY-1,j+N); 
	int k, l, m, n;
	/*i0,j0 es la esquina de arriba de la submatriz que queremos con respecto la matriz, 
	e i1 y j1 la esquina de abajo incluida esa posicion. */
	//printf("%d", i0);
	//printf("%d", j0);
	//printf("%d", i1);
	//printf("%d\n", j1);

	(*lenghtSubmatrizX)=i1-i0+1;
	(*lenghtSubmatrizY)=j1-j0+1;
	//printf("%d", (*lenghtSubmatrizX));
	//printf("%d\n", (*lenghtSubmatrizY));
	
	(*subMatriz) = ( double** )malloc((*lenghtSubmatrizX)*sizeof( double* ));
	for ( k=0; k<(*lenghtSubmatrizX); k++ )
	{
		(*subMatriz)[k] = (double*)malloc((*lenghtSubmatrizY)*sizeof(double));
	}
	
	/*Esta matriz puede ser de multiples dimensiones dependiendo de N y donde este i,j 
	(bordes y cuadros que se salen del borde)*/      
    for(k=i0, m=0;k<i0+(*lenghtSubmatrizX);k++, m++){  //relleno del resultado                               
            for(l=j0, n=0;l<j0+(*lenghtSubmatrizY);l++, n++){
                  (*subMatriz)[m][n] = *(double*)PyArray_GETPTR2(matriz,k,l);
				  //printf("%.10f      ",(*subMatriz)[m][n]);
            }
			//printf("\n");
    } 
	
}

double var (double** matriz, int lenghtMatrizX, int lenghtMatrizY)
{
	/*Esta funci�n calcula var = mean(abs(x - x.mean())**2) donde mean() es la media es decir la suma de las
	componentes dividido por el n�mero de elementos*/

      //calculo la media
	  double sumaTotal=0.0;
	  double mediaMatriz;
	  int i,j;
	  for(i=0;i<lenghtMatrizX;i++){  //relleno del resultado                               
            for(j=0;j< lenghtMatrizY;j++){
                  sumaTotal= sumaTotal + matriz[i][j];
            }
      }
	  
	  mediaMatriz=sumaTotal/(lenghtMatrizX*lenghtMatrizY);
	  //printf("%.10f media matriz\n",mediaMatriz);
	  
	  //resto la media a cada posici�n del vector, hago el valor absoluto y lo elevo a la 2.
	  for(i=0;i<lenghtMatrizX;i++){  //relleno del resultado                               
            for(j=0;j< lenghtMatrizY;j++){
                matriz[i][j]=pow(fabs(matriz[i][j]-mediaMatriz),2);
	        }
      }
	  
	  //vuelvo a calcular la media
	  sumaTotal=0.0;
	  for(i=0;i<lenghtMatrizX;i++){  //relleno del resultado                               
            for(j=0;j< lenghtMatrizY;j++){
                  sumaTotal= sumaTotal + matriz[i][j];
            }
      }
	  
	  //printf("%.10f resultado total\n",sumaTotal/(lenghtMatrizX*lenghtMatrizY));
	  return sumaTotal/(lenghtMatrizX*lenghtMatrizY);

}

void fminMatrix (PyArrayObject* m1, PyArrayObject* m2)
{
	/*Las dos matrices tienen el mismo tama�o mirar documentaci�n web numpy.fmin
	Calcula en m1 los valores minimos entre m1 y m2*/
	
	int lenghtMatrizX = PyArray_DIMS(m1)[0];
	int lenghtMatrizY = PyArray_DIMS(m1)[1];
	int i,j;

	for(i=0;i<lenghtMatrizX;i++){  //relleno del resultado                               
        for(j=0;j<lenghtMatrizY;j++){
            (*(double*)PyArray_GETPTR2(m1,i,j))= fmin(*(double*)PyArray_GETPTR2(m1,i,j),*(double*)PyArray_GETPTR2(m2,i,j));
				  
        }
    }	

}
 
//funcion final
static PyObject *waveletsC_lawmlNC(PyObject *self,PyObject *args) {
	/*
	Return the denoised image using NxN LAW-ML. Estimate the variance of original noise-free image for each wavelet
    coefficients using the MAP estimation for 4 sizes of square NxN neighborhood for N=[3,5,7,9].
	
	Tiene 6 par�metros. Primero la imagen con ruido, segundo y tercero arrays auxiliares inicializados a 0 del tama�o de la imagen con ruido, 
	cuarto parametro se devuelve la imagen sin ruido, quinto parametro sigma al cuadrado, sexto si se quiere utilizar multiples vecinos (N= 5,7,9)
	
	*/
	PyObject *imObjectRuido,*imObjectS,*imObjectT,*imObjectSinRuido;
	double sig0;
	int multipleNeighbour, i, j, lx, ly, k, l;
	double** mresult;
	PyArrayObject *imagenRuido, *imagenSinRuido, *S, *T;
	PyArg_ParseTuple(args,"OOOOdi",&imObjectRuido, &imObjectS, &imObjectT, &imObjectSinRuido, &sig0, &multipleNeighbour);
	imagenRuido = (PyArrayObject *)imObjectRuido;
	imagenSinRuido = (PyArrayObject *)imObjectSinRuido;
	S= (PyArrayObject *)imObjectS;
	T= (PyArrayObject *)imObjectT;
	
		
	/*sig0 = SIGMA*SIGMA ya viene calculado de fuera
    N = (5,7,9)
    (m0,m1) = X.shape
    S = numpy.zeros((m0,m1))*/
	 
	//double N[3] = {5,7,9}; al pasarlo a C se puede hacer con un for 5, 7, 9
    int m0 = PyArray_DIMS(imagenRuido)[0];
	int m1 = PyArray_DIMS(imagenRuido)[1];
	
	/*
	#calculando  para tama�o N=3 para tenerlo como m�nimo
    for x in xrange(m0):
        for y in xrange(m1):
            sub = subm(X,3,x,y)
            S[x,y] = max(0, numpy.var(sub) - sig0 )
	*/
	//calculando  para tama�o N=3 para tenerlo como m�nimo
	for(i=0;i<m0;i++){                                
        for(j=0;j<m1;j++){
			mresult=NULL;
			submatrix(imagenRuido,3,i,j,&mresult,&lx,&ly);
			/*printf("Submatriz de %i %i     ",lx, ly);
			for(x=0;x<lx;x++){                               
				for(y=0;y<ly;y++){
					  printf("%.10f      ",mresult[x][y]);
				}
				printf("\n");
			}
			printf("\n");*/
			*(double*)PyArray_GETPTR2(S,i,j)=fmax(0, var(mresult,lx,ly)-sig0);
			for ( k=0; k<lx; k++ )
			{
				free(mresult[k]);
			}
			free(mresult);
			//printf("%.10f   ",*(double*)PyArray_GETPTR2(S,i,j));
        }
		//printf("\n");
    } 
	
	
	/*if MULTIPLE_NEIGHBOR:
        #calculando  para el restode los tama�os N=5,7,9
        T = numpy.zeros((m0,m1))
        for n in N:
            for x in xrange(m0):
                for y in xrange(m1):
                    sub = subm(X,n,x,y)
                    T[x,y] = max(0, numpy.var(sub) - sig0 )

            S = numpy.fmin(S,T)*/

	if (multipleNeighbour==1)//true
	{

		for (k=5;k<11;k=k+2) //recorre N=5,7,9
		{
			for(i=0;i<m0;i++){                                
				for(j=0;j<m1;j++){
					mresult=NULL;
					submatrix(imagenRuido,k,i,j,&mresult,&lx,&ly);
					/*printf("Submatriz de %i %i     ",lx, ly);
					for(x=0;x<lx;x++){                               
						for(y=0;y<ly;y++){
							  printf("%.10f      ",mresult[x][y]);
						}
						printf("\n");
					}
					printf("\n");*/
					*(double*)PyArray_GETPTR2(T,i,j)=fmax(0, var(mresult,lx,ly)-sig0);
					for ( l=0; l<lx; l++ )
					{
						free(mresult[l]);
					}
					free(mresult);
					//printf("%.10f   ",*(double*)PyArray_GETPTR2(T,i,j));
				}
				//printf("\n");
			}
		fminMatrix(S,T);
		}
	}	
		
    //return X*S/(S+sig0)
	for(i=0;i<m0;i++){                                
        for(j=0;j< m1;j++){
			*(double*)PyArray_GETPTR2(imagenSinRuido,i,j)=((*(double*)PyArray_GETPTR2(imagenRuido,i,j))*(*(double*)PyArray_GETPTR2(S,i,j)))/((*(double*)PyArray_GETPTR2(S,i,j))+sig0);
        } 
	}
	
	return Py_BuildValue("i",0);
}	 



static PyMethodDef waveletsCMethods[] = {
 {"lawmlNC", waveletsC_lawmlNC, METH_VARARGS, "Extracci�n de ruido de imagen"},
 {NULL, NULL, 0, NULL}
 };

void initwaveletsC(void){
 (void) Py_InitModule("waveletsC", waveletsCMethods);
 }
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 



