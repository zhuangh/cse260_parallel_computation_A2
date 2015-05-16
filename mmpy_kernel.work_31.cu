// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

#define VOLV
// #define instruction
#define As(i,j) As[i][j]
#define Bs(i,j) Bs[i][j]

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
/*
#ifdef USE_SHARED_UNCOAL
//     int N =  blockDim.y*gridDim.y;


    const unsigned int bx = BLOCKDIM_X, by=BLOCKDIM_Y;
    const unsigned int tx = threadIdx.x , ty = threadIdx.y;
    const unsigned int I = blockIdx.y * by +ty, J = blockIdx.x *bx +tx;
    const unsigned int gx = gridDim.x, gy = gridDim.y;
    __shared__ _DOUBLE_  a[BLOCKDIM_X][BLOCKDIM_Y], b[BLOCKDIM_X][BLOCKDIM_Y];
    if((I<N)&&(J<N)){
	_DOUBLE_ c = 0.0;
	for(unsigned int k = 0 ; k < gy; k++){
	    a[ty][tx] = A[I*N +k*bx + tx];
	    b[ty][tx] = B[J+N*(k*bx+ty)];
	    __syncthreads();
	    for(unsigned int kk = 0; kk<bx; kk++)
		c+=a[ty][kk]*b[kk][tx];
		// may use atomaticadd
	    __syncthreads();
	}
    
    C[I*N+J] = c;
    }
#elif defined VOLV
    
k  

  */
 
    int block_size = BLOCKDIM_X;
    int wA = N;
    int wB = N;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const unsigned int BLOCK_SIZE = BLOCKDIM_X;
    
    int BLK_NUM = 1 + N / BLOCKDIM_X ; 

    if( N % BLOCKDIM_X  ==0 ){

	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd = aBegin +wA -1;
	int aStep = BLOCK_SIZE;

	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * wB;


	_DOUBLE_ Csub = 0.;

	for( int a = aBegin, b = bBegin; a <= aEnd ; a+=aStep, b+=bStep){
	    __shared__ _DOUBLE_  As[BLOCK_SIZE][BLOCK_SIZE ];	
	    __shared__ _DOUBLE_  Bs[BLOCK_SIZE][BLOCK_SIZE];	
	    As(ty, tx) = A[a + wA*ty +tx];
	    Bs(ty, tx) = B[b + wB*ty +tx];
	    __syncthreads();

#ifdef instruction

// #pragma unroll
	    for (int k = 0 ; k < BLOCK_SIZE ; k=k+2)
	    {
		Csub += (As(ty,k) * Bs(k, tx) + As(ty,k+1)*Bs(k+1, tx) ) ;

	    }
	    __syncthreads();

#else
// #pragma unroll
	    for (int k = 0 ; k < BLOCK_SIZE ; ++k)
	    {
		Csub += As(ty,k) * Bs(k, tx);

	    }
	    __syncthreads();
#endif
	}
	int c = wB* BLOCK_SIZE * by + BLOCK_SIZE *bx;
	C[c+ wB *ty + tx] =Csub;
    }
    else{ 
#ifdef VOLV
	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd = aBegin +wA -1;
	int aStep = BLOCK_SIZE;

	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * wB;


	_DOUBLE_ Csub = 0.;
	int step_no = 0;
	for( int a = aBegin, b = bBegin ;
	     a <= aEnd ;
	     a+=aStep, b+=bStep, step_no++){

	    __shared__ _DOUBLE_  As[BLOCK_SIZE][BLOCK_SIZE ];	
	    __shared__ _DOUBLE_  Bs[BLOCK_SIZE][BLOCK_SIZE];	
	    //  if( by * BLOCK_SIZE + ty < N && bx * BLOCK_SIZE + bx < N )
	    if ( a - aBegin + tx < N
		 && by * BLOCK_SIZE + ty < N )
		As(ty, tx) = A[a + wA*ty +tx];
	    else
		As(ty, tx) = 0.0;

	    // if( by * BLOCK_SIZE + ty < N && bx * BLOCK_SIZE + bx < N )
	    if (  step_no * BLOCK_SIZE + ty < N
		 && bx * BLOCK_SIZE +tx < N
	       )
		Bs(ty, tx) = B[b + wB * ty +tx];
	    else 
		Bs(ty , tx) =0.0;

	    __syncthreads();
#pragma unroll
	    for (int k = 0 ; k < BLOCK_SIZE ; ++k)
		Csub += As(ty,k) * Bs(k, tx);
	    __syncthreads();

	}

	int c = N * BLOCK_SIZE * by + BLOCK_SIZE *bx;

//	if( by * BLOCK_SIZE + ty < N && bx * BLOCK_SIZE + tx < N )
//	if ( c +wB*ty+tx < N*N )
	if( bx * BLOCK_SIZE + tx < N && by * BLOCK_SIZE + ty < N){
	    C[c+ wB *ty + tx] =Csub;
	}


#else
	int _row = by * BLOCK_SIZE + ty;
	int _col = bx * BLOCK_SIZE + tx;

	_DOUBLE_ Csub = 0;

	__shared__ _DOUBLE_ As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ _DOUBLE_ Bs[BLOCK_SIZE][BLOCK_SIZE];

	for (int i = 0; i < (int)(ceil ((float) N / BLOCK_SIZE)); ++i){

	    if (i * BLOCK_SIZE + tx < N && _row < N)
		As(ty, tx) = A[_row * N + i * BLOCK_SIZE + tx];
	    else
		As(ty, tx) = 0.0;

	    if (i * BLOCK_SIZE + ty < N && _col < N)
		Bs(ty, tx) = B[(i * BLOCK_SIZE + ty) * N + _col];
	    else
		Bs(ty, tx) = 0.0; 

	    __syncthreads();

	    for (int j = 0; j < BLOCK_SIZE; ++j){

		Csub += As(ty, j) * Bs(j, tx);
	    }

	    __syncthreads();
	}	

	if (_row < N && _col < N){
	    C[_row * N + _col] = Csub;
	}
#endif

}
	// zhuangh's note:
	// Jin put something below , ensure the odd situation  
	/*
	   int aBegin = wA * BLOCK_SIZE * by;
	   int aEnd = aBegin + wA -1;
	   int aStep = BLOCK_SIZE;

	   int bBegin = BLOCK_SIZE * bx;
	   int bStep = BLOCK_SIZE * wB;

	   int BLK_XSZ; 
	   int BLK_YSZ ;
	   BLK_XSZ =  BLK_YSZ = BLOCK_SIZE ;

	   _DOUBLE_ Csub = 0.;

	   BLK_XSZ = N - (BLK_NUM-1 ) * BLOCK_SIZE ;
	   BLK_YSZ = N - (BLK_NUM-1 ) * BLOCK_SIZE ;

	   for( int a = aBegin, b = bBegin; a <= aEnd ; a+=aStep, b+=bStep){

	   __shared__ _DOUBLE_  As[BLOCK_SIZE][BLOCK_SIZE];	
	   __shared__ _DOUBLE_  Bs[BLOCK_SIZE][BLOCK_SIZE];	


	   if( ( aBegin <= (BLK_NUM-2) * BLOCK_SIZE) 
	   && ( bBegin <= (BLK_NUM - 2 )*BLOCK_SIZE  )){        
	   As(ty, tx) = A[a + wA*ty +tx];
	   Bs(ty, tx) = B[b + wB*ty +tx];

	   __syncthreads();
#pragma unroll
for (int k = 0 ; k < BLOCK_SIZE ; ++k)
Csub += As(ty,k) * Bs(k, tx);
__syncthreads();


}
else if( aBegin <= (BLK_NUM-2) * BLOCK_SIZE 
&& bBegin > (BLK_NUM-2) * BLOCK_SIZE ){
if(tx < BLK_XSZ ){
As(ty, tx) = A[a + wA*ty +tx];
Bs(ty, tx) = B[b + wB*ty +tx];

__syncthreads();
#pragma unroll
for (int k = 0 ; k < BLK_XSZ ; ++k)
Csub += As(ty,k) * Bs(k, tx);
__syncthreads();


}
}
else if ( aBegin <= (BLK_NUM-2) * BLOCK_SIZE 
&& bBegin > (BLK_NUM-2) * BLOCK_SIZE ){
if(ty < BLK_YSZ){
As(ty, tx) = A[a + wA*ty +tx];
Bs(ty, tx) = B[b + wB*ty +tx];

__syncthreads();
#pragma unroll
for (int k = 0 ; k < BLOCK_SIZE ; ++k)
Csub += As(ty,k) * Bs(k, tx);
__syncthreads();


}
}
else{
if(ty < BLK_YSZ && tx < BLK_XSZ  ){
As(ty, tx) = A[a + wA*ty +tx];
Bs(ty, tx) = B[b + wB*ty +tx];

	__syncthreads();
#pragma unroll
	for (int k = 0 ; k < BLK_XSZ ; ++k)
	    Csub += As(ty,k) * Bs(k, tx);
	__syncthreads();

    }

    }

}

int c = wB* BLOCK_SIZE * by + BLOCK_SIZE *bx;  // wrong access 
if(c+ wB *ty + tx < N*N  )
    C[c+ wB *ty + tx] =Csub;


    */


// may use atomaticadd
/*
#else

int I =  blockIdx.x*blockDim.x + threadIdx.x;
int J =  blockIdx.y*blockDim.y + threadIdx.y;

    if((I < N) && (J < N)){
        _DOUBLE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _DOUBLE_ a = A[I * N + k];
            _DOUBLE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
#endif
    */
}
