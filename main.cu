#include<stdio.h>
#include "a.cuh"
#include "GPU.cuh"
#include "myqr.cuh"




__host__ void print_help (int exit_code) {


    printf("  -h    print this help and exit\n");
    printf("  -r    provide the number of rows\n");
    printf("  -c    provide the number of colums\n\n");
    printf("  -s    starting value of iteration\n\n");
    printf("  -p    no. of permutation in 1 GPU\n\n");
    printf("  -g    Total no. of GPU\n\n");

    printf("  Example: ./qr_gpu -r 800 -c 600 -s 0 -p 1000 -g 6 \n\n");

    exit_code == -1 ? exit(EXIT_FAILURE) : exit(EXIT_SUCCESS);
}

__host__ int option_parser (int argc, char **argv, int * m, int * n, int *start, int *permutation, int *gpu) {

    int opt;

    if (argc < 5) {
        fprintf(stderr, "The program needs arguments...\n\n");
        print_help(1);
    }

    opterr = 0;

    while ( -1 != (opt = getopt (argc, argv, "hr:c:s:p:g:"))) {
        switch (opt) {
            case 'h': 
                print_help(0);
            case 'r': 
                if ((*m = atoi(optarg)) < 2) return -1; 
                break;
            case 'c': 
                if ((*n = atoi(optarg)) < 2 || *n > *m) return -1; 
                break;
            case 's': 
                *start = atoi(optarg);
                break;
            case 'p': 
                *permutation = atoi(optarg);
                break;
            case 'g': 
                *gpu = atoi(optarg);
                break;
            case '?':
                if (optopt == 'r' || optopt == 'c'|| optopt == 's' || optopt == 'p' || optopt == 'g')
                    fprintf(stderr,"Option -%c requires an argument.\n",optopt);
                else if (isprint (optopt))
                    fprintf(stderr,"Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,"Unknown option chr `\\x%x'.\n", optopt);
                return -1;
            default:
                fprintf(stderr, "default switch-case statement reached\n");
                return -1;
        }

    }
    return 0;
}





void loop(float **arr,float *a_d, float *A_d, float *y_d, float *res_d, float *res, float *R_d, float *Q_d, float *A1_d, float *y1_d, float *coef_d, float *x_d, float *z_d, float *u_d, float *Qvv_d, float *Avv_d, float *p_d, int m, int n, int p){

    int i=0;

    int f2=0;
    if (m*n>128){ f2 = m*n/128;}
    dim3 dimBlock_A(m/f2+1,n);
    dim3 dimGrid_A(f2+1,1);

    int f3=0;
    if (m*m>256) {f3 = m*m/256;}
    dim3 dimBlock_Q(m/16,m/16);
    dim3 dimGrid_Q(f3+1,f3+1);

    int f4=0; int f5=0;
    f4 = p/32; f5=m/16;
    dim3 dimBlock_sq(16,32);
    dim3 dimGrid_sq(f5+1,f4+1);




    for (i=0;i<7802;i++){



        Take_A <<< dimGrid_A, dimBlock_A >>> (a_d, A_d, i, m, n);
        form_Q <<< dimGrid_Q, dimBlock_Q >>> (Q_d,m);
        qr( A_d, y_d, res_d, R_d, Q_d, A1_d, y1_d, coef_d, x_d, z_d, u_d, Qvv_d, Avv_d, p_d, m, n, p);

        // square of residual
        square <<< dimGrid_sq, dimBlock_sq >>> (res_d, m, p);

        cudaMemcpy(res, res_d, m*p*sizeof(float), cudaMemcpyDeviceToHost);
               
        // sum of residual
        for(int j=0; j<p; j++){
            float sum=0;
            for (int k=0; k<m; k++){
                sum+=res[k*p + j];
            }
            arr[j][i]=sum;
        }


    } 


    return;
}

void minimum( float **array, FILE *file, int n, int p){
    
    float a;
    for(int i=0; i<p; i++){
        a = array[i][0];
        for ( int j=1; j<n ; j++){
            if (array[i][j]<a) a = array[i][j];
        }
        fprintf(file,"%f\n", a);
        fflush(file);
    }
    return;

}



int main(int argc, char **argv) {

    // Intializing variables
    int m,n,s,p,g;
    if (0 != option_parser(argc, argv, &m, &n, &s, &p, &g)) {
        fprintf(stderr, "Can\'t continue, exiting now!\n"); 
        exit(EXIT_FAILURE);
    }

    // Initializing All thw memory variables

    // Taking input of y and a
    float *Y = (float *) malloc(m * 1 * sizeof(float));
    float *y = (float *) malloc(m * p * sizeof(float));
    float *a = (float *) malloc(7802* m * n * sizeof(float));
    float *sample = (float *) malloc( m * 1000 * sizeof(float));
    input(Y, m, "pheno.txt");
    input(a, 7802*m*n, "genoT.txt");
    input(sample, 1000*m, "sampling.txt");

    float *a_d;    
    cudaMalloc(&a_d, 7802*m*n*sizeof(float));
    cudaMemcpy(a_d, a, 7802*m*n*sizeof(float), cudaMemcpyHostToDevice);

    float *sample_d;    
    cudaMalloc(&sample_d, m*1000*sizeof(float));
    cudaMemcpy(sample_d, sample, m*1000*sizeof(float), cudaMemcpyHostToDevice);

    float *Y_d;            
    cudaMalloc(&Y_d, m*1*sizeof(float));
    cudaMemcpy(Y_d, Y, m*1*sizeof(float), cudaMemcpyHostToDevice);


    // Array which takes the sum of array of each 7802 matrix
    float **arr = (float **) malloc(p * sizeof(float*));
    for (int k = 0; k < p; k++) {
        arr[k] = (float *) malloc(sizeof(float) * 7802);
    }
    float *lrs = (float *) malloc(p * sizeof(float));
    //cudaMalloc(&arr_d, 7802 * 1 * sizeof(float));

    float *res = (float *) malloc(m * p * sizeof(float));
    float *res_d;
    cudaMalloc(&res_d, m*p*sizeof(float));
    float *A_d;         
    cudaMalloc(&A_d, m*n*sizeof(float));
    float *y_d;            
    cudaMalloc(&y_d, m*p*sizeof(float));

    float *R_d; 
    cudaMalloc(&R_d, m*n*sizeof(float));
    float *Q_d;            
    cudaMalloc(&Q_d, m*m*sizeof(float));
    float *A1_d; 
    cudaMalloc(&A1_d, m*n*sizeof(float));
    float *y1_d; 
    cudaMalloc(&y1_d, n*p*sizeof(float));
    float *coef_d; 
    cudaMalloc(&coef_d, n*p*sizeof(float));
    float *p_d; 
    cudaMalloc(&p_d, 1*p*sizeof(float));         


    float *x_d;
    cudaMalloc(&x_d, (m)*1*sizeof(float));
    float *z_d;
    cudaMalloc(&z_d, (n)*1*sizeof(float)); 
    float *u_d;
    cudaMalloc(&u_d, m*1*sizeof(float));
    float *Qvv_d;
    cudaMalloc(&Qvv_d, (m)*(m)*sizeof(float));
    float *Avv_d;
    cudaMalloc(&Avv_d, (m)*(n)*sizeof(float));


    FILE * file;
	file = fopen("output.txt", "a");

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    int f4=0; int f5=0;
    f4 = p/32; f5=m/16;
    dim3 dimBlock_y(16,32);
    dim3 dimGrid_y(f5+1,f4+1);


    /* SAMPLING */
    update_y <<< dimGrid_y,dimBlock_y >>> (y_d, Y_d, sample_d, m, n, s, p);      
    loop(arr, a_d, A_d, y_d, res_d, res, R_d, Q_d, A1_d, y1_d, coef_d, x_d, z_d, u_d, Qvv_d, Avv_d, p_d, m, n, p);
    minimum(arr, file, 7802, p);
    fclose(file);

    

    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Elapsed time of GPU %d : %f ms\n" , g, elapsedTime);


 
    
    return 0;
}





