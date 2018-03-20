#include<stdio.h>

__global__ void Take_A(float *d_a, float *d_A, int i, int m, int n) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    
    if(tx<m && ty<n ){
        d_A[ty + tx*n] = d_a[ty + tx*n + i*m*n];
    }

        
}

__global__ void copy(float *d_A, float *d_B, int Mrow, int Mcl) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    d_A[tx*Mcl + ty] = d_B[tx*Mcl + ty];
        
}

__global__ void update_y(float *d_y, float *d_Y, float *sample, int m, int n, int i, int p) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if(tx<m && ty<p){
        d_y[tx*p + ty] = d_Y[int(sample[(i+ty)*m + tx])-1];
    }
        
}

__global__ void square (float * a, int m, int p){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx<m && idy<p) a[idx*p +idy] = a[idx*p + idy]*a[idx*p + idy];

}

__global__ void form_Q(float * d_Q,const int m) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < m) && (j < m)) {
        if(i==j){
            d_Q[i + i*m] = 1;
        }else{
            d_Q[i + j*m] = 0;
        }
    }
    
}

/** DEVICE FUNCTIONS FOR CALCULATION OF COEFFICIENTS OF AX=Y EQUATION **/


__global__ void form_r (float *R, float *r, int m, int n, int k) {

    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j > k) {
        r[j-k-1] = R[(k)*n + j];

    }



}

__global__ void sum_rx (float *R, float *x, int m, int n, int i , float* pvalue, int p) {
    // get x,y cordinates
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int y = blockIdx.y * blockDim.y + threadIdx.x;
    pvalue[tx]=0;
    for(int j=i+1; j<=n-1; j++){
        pvalue[tx]+=R[i*n + j]*x[tx + (j)*p];
      
    } 

}


__global__ void crossprod_Qty(float *d_M, float *d_N, float *d_P, int row, int Mcl, int p) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    //Pvalue stores the d_P element that is computed by the thread
    float Pvalue = 0;
    
    if (tx<Mcl && ty<p){
        for(int k = 0; k < row ; ++k) {
            float Mdelement = d_M[k*row + tx];
            float Ndelement = d_N[k*p + ty];
            Pvalue += (Mdelement*Ndelement);
            }
        d_P[tx*p + ty] = Pvalue;
    }


}


__global__ void update_coef (float *coef, float *y, float *R, float* pvalue, int n, int i, int p) {

    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = R[i*n+i];
    if(tx<p){
        coef[i*p + tx] = (y[i*p + tx] - pvalue[tx])/a;
    }

}





/** DEVICE FUNCTIONS FOR CALCULATION OF RESIDUAL OF AX=Y EQUATION **/


__global__ void prod(float *d_M, float *d_N, float *P, float *Y, int Mrow, int Mcl, int p) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;

    //Pvalue stores the d_P element that is computed by the thread
    for(int i=0;i<p;i++){
        float Pvalue=0;
        if (tx<Mrow){
            for(int k = 0; k < Mcl ; ++k) {
                float Mdelement = d_M[tx*Mcl + k];
                float Ndelement = d_N[i + k*p];
                Pvalue += (Mdelement*Ndelement);
            }
            P[tx*p + i] = Y[tx*p + i] - Pvalue;
        }
    }

    
}







/** DEVICE FUNCTIONS FOR CALCULATION OF Q AND R S.T A=QR **/


__global__ void update_x (float *R, float *x, int m, int n, int k) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < m-k) {
        x[j] = R[(j+k)*n + k];
    }
}



float norm (float *y, int m) { 

    float z; z=0;    
    for ( int j=0; j<m; j++){
        z= z + y[j]*y[j];
    }
    
    float s; s = sqrt(z);
    return s;

    
}


__global__ void w (float *ar, float *w, float s, int m, int k) {
    // block colum * block dim + column (computed by each thread)
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // and scale
    if (i<m-k) w[i] = ar[i]*s;
}




__global__ void crossprod_Atv(float *d_M, float *d_N, float *d_P, int Mrow, int Mcl, int Ncl, int i) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;


    float Pvalue = 0;

    if (tx<Mcl-i){
        for(int k = 0; k < Mrow-i ; ++k) {
            float Mdelement = d_M[(k+i)*Mcl + tx + i];
            float Ndelement = d_N[k*Ncl];
            Pvalue += (Mdelement*Ndelement);
            //printf("%d*%d\n",Mdelement, Ndelement);

            }

        d_P[tx*Ncl] = Pvalue;
    }

    
}


__global__ void Qn(float *d_u,float *d_x, float *Qvv, float s, int Mrow, int k) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0;

    if (tx<Mrow && ty<Mrow-k){

        float Mdelement = d_u[tx];
        float Ndelement = d_x[ty];
        Pvalue = (Mdelement*Ndelement);

        Qvv[tx*(Mrow-k) + ty] = Pvalue*s;
            
    }

    
    
}

__global__ void An(float *d_x,float *d_y, float *Avv, float s, int Wrow, int Yrow, int k) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0;

    if (tx<Wrow-k && ty<Yrow-k){

        float Mdelement = d_x[tx];
        float Ndelement = d_y[ty];
        Pvalue = (Mdelement*Ndelement);

        Avv[tx*(Yrow-k) + ty] = Pvalue*s;
            
    }

    
    
}

__global__ void u(float *d_M, float *d_N, float *P, int Mrow, int Mcl, int Nrow, int Ncl, int i) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;

    //Pvalue stores the d_P element that is computed by the thread
    float Pvalue = 0;

    if (tx<Mrow){
        for(int k = 0; k < Mcl-i ; ++k) {
            float Mdelement = d_M[tx*Mcl + k + i];
            float Ndelement = d_N[k*Ncl];

            Pvalue += (Mdelement*Ndelement);
            }

        P[tx*Ncl] = Pvalue;
    }

    
}



__global__ void uu(float *d_M, float *d_N, float *P, float s, int m, int n, int i) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    //Pvalue stores the d_P element that is computed by the thread
    float Pvalue1 = 0;
    float Pvalue2 = 0;
    __shared__ float u1[83];
    __shared__ float w1[83];

    if (tx<m && ty<1){
        for(int k = 0; k < m-i ; ++k) {
            float Mdelement = d_M[tx*m + k + i];
            float Ndelement = d_N[k*1 + ty];
            //printf("%f ", Mdelement);
            Pvalue1 += (Mdelement*Ndelement);

            }

        u1[tx*1 + ty] = Pvalue1;
        //printf("%f ... ", Pvalue1);

    }



    __syncthreads();


   if (ty<m-i && tx<1) {
        w1[ty] = d_N[ty]*s;
    }

    __syncthreads();


    if (tx<m && ty<m-i){

        float Mdelement1 = u1[tx];
        float Ndelement1 = w1[ty];
        Pvalue2 = (Mdelement1*Ndelement1);

        P[tx*(m-i) + ty] = Pvalue2;
        printf("%f  ", Pvalue2);
            
    }




    
}


__global__ void update_Q(float *d_Q, float *Qvv, int Mrow, int Mcl, int i) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx<Mrow && ty<Mcl){
        d_Q[tx*Mrow + ty + i] = d_Q[tx*Mrow + ty +i] - Qvv[tx*Mcl + ty];
    }    
}

__global__ void update_A(float *d_A, float *Avv, int Mrow, int Mcl, int i) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx<Mrow-i && ty<Mcl-i){
        d_A[(tx+i)*Mcl + ty + i] = d_A[(tx+i)*Mcl + ty +i] - Avv[tx*(Mcl-i) + ty];
    }    
}

__global__ void update_R(float *d_A, float *d_R, int row, int cl) {
    //Thread Indx in 2D
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx<row && ty<cl){
        if (tx<=ty){
            d_R[tx*cl + ty] = d_A[tx*cl + ty];
        }else{
            d_R[tx*cl + ty] = 0;
        }
    }    
}











__host__ void gpuAssert(cudaError_t code, char *file, int line) {

    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", 
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__host__ void print_matrix (float * matrix, int m, int n, int ld) {

    int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            printf("%f ", matrix[i*ld + j]);
        printf("\n");
    }
}


