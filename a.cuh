#include <stdio.h>
__host__ void printMat(float *, int, int, int);
void input(float *matlino, int n, char *filename)
{



    //Open linogram
    FILE * flino;
    flino = fopen (filename,"r");      //also tried with "rb" instead of "r"
    if (flino!=NULL)
    {
        int i;
        for(i = 0; i < n; i++){
            fscanf(flino, "%f", &matlino[i]);
            

        }
        fclose(flino);
    }
    else
        puts("impossivel abrir linograma");


    return;
}

__host__ void printMat (float * matrix, int m, int n, int ld) {

    int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            printf("%f ", matrix[i*ld + j]);
        printf("\n");
    }
}
