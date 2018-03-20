// Libraries
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

// Host code
void qr(float *A_d, float *y_d, float *res_d, float *R_d, float *Q_d, float* A1_d, float *y1_d, float *coef_d, float *x_d, float *z_d, float *u_d, float *Qvv_d, float *Avv_d, float *p_d, int m, int n, int p) {

    int i;



    //*** INVOKE KERNEL ***//

    int f1;
    if (m<512) f1=m; else f1=512;
    dim3 dimBlock_B (m, 1);
    dim3 dimGrid_B (f1/m, 1);

    int f2=0;
    if (m*n>128){ f2 = m*n/128;}
    dim3 dimBlock_R(m/f2+1,n);
    dim3 dimGrid_R(f2+1,1);

    int f3=0;
    if (m*m>256) {f3 = m*m/256;}
    dim3 dimBlock_Q(m/16,m/16);
    dim3 dimGrid_Q(f3+1,f3+1);

    int f4=0;
    if (n*n>256) {f4 = n*n/256;}
    dim3 dimBlock_R1(n/16,n/16);
    dim3 dimGrid_R1(f4+1,f4+1);

    int f5=0;
    if (m*n>128){ f5 = m*n/128;}
    dim3 dimBlock_Q1(m/f5,n);
    dim3 dimGrid_Q1(f5+1,1);

    int f6;
    if (n<512) f6=n; else f6=512;
    dim3 dimBlock_B1 (n, 1);
    dim3 dimGrid_B1 (f6/n, 1);

    int f7=0;
    f7 = p/32; 
    dim3 dimBlock_Qty(8,32);
    dim3 dimGrid_Qty(1,f7+1);
    
    


    //*** COMPUTING A=QR BY HOUSEHOLDER METHOD ***//


    copy <<< dimGrid_Q1, dimBlock_Q1 >>> (A1_d, A_d, m, n);

    
    for (i = 0; i < n; i++) {

        // initialising



        float *x_h = (float *) malloc((m-i)*sizeof(float)); 
        //float *Qvv_h = (float *) malloc(Qvv_size); 
        //float *Avv_h = (float *) malloc(Avv_size); 

        
        // Step #1 --> x = R(i:m,i)
        update_x <<< 1, m-i >>> (A_d, x_d, m, n, i);

        // Step #2 --> norm(x)
        cudaMemcpy(x_h, x_d, (m-i)*sizeof(float), cudaMemcpyDeviceToHost); 
        float g=norm(x_h,m-i);

        // Step #3 --> v(1)=x(1)+g
        
        if (x_h[0]<0){
            x_h[0] = x_h[0] - g;
        }else{
           x_h[0] = x_h[0] + g;
        }
       

        // Step #4 --> beta = 2/square(norm(v))
        float s=norm(x_h,m-i);
        cudaMemcpy(x_d, x_h, (m-i)*1*sizeof(float), cudaMemcpyHostToDevice);
       
        float beta = 2/(s*s);

        // Step #5 --> update Q   
        u <<< dimGrid_B, dimBlock_B >>> (Q_d, x_d, u_d, m, m, m-i ,1 , i);
        Qn <<< dimGrid_Q, dimBlock_Q >>> (u_d, x_d, Qvv_d, beta, m, i);
        update_Q <<< dimGrid_Q, dimBlock_Q >>> (Q_d, Qvv_d, m, m-i, i);




        // Step #6 --> update A
        crossprod_Atv <<< 1, n-i >>> (A_d, x_d, z_d, m, n, 1, i);
        An <<< dimGrid_Q, dimBlock_Q >>> (x_d, z_d, Avv_d, beta, m, n, i);
        update_A <<< dimBlock_R, dimGrid_R >>> (A_d, Avv_d, m, n, i);
                 
        
    }

    // Copy the results from device memory to host memory


    update_R <<< dimGrid_R, dimBlock_R >>> (A_d, R_d, m, n);

    cudaFree(A_d);


    //*** SOLVING MATRIX Ax=y SO THAT I CAN FIND COEFFICIENTS ***//

    crossprod_Qty <<< dimGrid_Qty, dimBlock_Qty >>> (Q_d, y_d, y1_d, m, n, p);



    for (int i=n-1; i>=0; i--){

        // Step #2 --> sum(r(j)*x(j))
        sum_rx <<< 8,p/8+1 >>> (R_d, coef_d, m, n, i, p_d, p);
        //cudaFree(r_d);

        update_coef <<< 8,p/8+1>>> (coef_d, y1_d, R_d, p_d, n, i, p);  

                 

    }


    //*** ALLOCATE DEVICE MEMORIES FOR CALCULATION OF RESIDUAL ***//


    prod <<< 1, m >>> (A1_d, coef_d, res_d, y_d, m, n, p);
    

    return;
}

















