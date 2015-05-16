// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

#define As(i,j) As[i][j]
#define Bs(i,j) Bs[i][j]

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    int wA = N;
    int wB = N;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const unsigned int BLOCK_SIZE = BLOCKDIM_X;


    if(by < N/BLOCKDIM_X && N % BLOCKDIM_X  ==0 ){

	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd = aBegin +wA -1;
	int aStep = BLOCK_SIZE ;

	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * wB;


	_DOUBLE_ Csub[4] ={ 0.,0.,0.,0.};

	for( int a = aBegin, b = bBegin; a <= aEnd ; a+=aStep, b+=bStep){
	    __shared__ _DOUBLE_  As[BLOCK_SIZE ][BLOCK_SIZE+1 ];	
	    __shared__ _DOUBLE_  Bs[BLOCK_SIZE ][BLOCK_SIZE+1 ];	
	    As(tx, ty) = A[a + wA*ty +tx];
	    Bs(tx, ty) = B[b + wB*ty +tx];
	    As(tx, (ty + BLOCK_SIZE/4)) = A[ a+ wA *(ty+ BLOCK_SIZE/4) +tx];
	    Bs(tx, (ty + BLOCK_SIZE/4)) = B[ b+ wB *(ty+ BLOCK_SIZE/4) +tx];
	    As(tx, (ty + BLOCK_SIZE/2)) = A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];
	    Bs(tx, (ty + BLOCK_SIZE/2)) = B[ b+ wB *(ty+ BLOCK_SIZE/2) +tx];
	    As(tx, (ty + BLOCK_SIZE/4*3)) = A[ a+ wA *(ty+ BLOCK_SIZE/4*3) +tx];
	    Bs(tx, (ty + BLOCK_SIZE/4*3)) = B[ b+ wB *(ty+ BLOCK_SIZE/4*3) +tx];

	    __syncthreads();


#pragma unroll
	    for (int k = 0 ; k < BLOCK_SIZE; k++)
	    {
		Csub[0] += As(k,ty) * Bs(tx, k);
		Csub[1] += As(k, ty + BLOCK_SIZE/4) * Bs(tx, k)  ;
		Csub[2] += As(k, ty + BLOCK_SIZE/2) * Bs(tx, k)  ;
		Csub[3] += As(k, ty + BLOCK_SIZE/4*3) * Bs(tx, k)  ;

	    }
	    __syncthreads();

	}

	int c = wB* BLOCK_SIZE * by + BLOCK_SIZE *bx;
	C[c+ wB *ty + tx] =Csub[0];
	C[c+ wB *(ty+ BLOCK_SIZE/4) + tx] = Csub[1];
	C[c+ wB *(ty+ BLOCK_SIZE/2) + tx] = Csub[2];
	C[c+ wB *(ty+ BLOCK_SIZE/4*3) + tx] = Csub[3];
    }

    if (by < N / BLOCKDIM_X + 1 && N % BLOCKDIM_X  !=0 ){

	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd = aBegin +wA - 1;
	int aStep = BLOCK_SIZE;

	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * wB;

	_DOUBLE_ Csub[4] ={ 0.,0.,0.,0.};

	int step_no = 0;

	int aflag = 0; 
	int bflag = 0;

	for(int a = aBegin, b = bBegin;
	     a <= aEnd;
	     a+=aStep, b+=bStep, step_no++){

	    __shared__ _DOUBLE_  As[BLOCK_SIZE][BLOCK_SIZE +1 ];	
	    __shared__ _DOUBLE_  Bs[BLOCK_SIZE][BLOCK_SIZE +1 ];	
	    //  if( by * BLOCK_SIZE + ty < N && bx * BLOCK_SIZE + bx < N )
	    aflag = a - aBegin + tx;     
	    bflag = by * BLOCK_SIZE +ty ;

	    if ( aflag < N && bflag  < N )  As(tx, ty) = A[a + wA*ty +tx];
	    else As(tx, ty) = 0. ;

	    if(  bflag +  BLOCK_SIZE/4 < N && aflag  < N  ) 
		As(tx, (ty + BLOCK_SIZE/4)) = A[ a+ wA *(ty+ BLOCK_SIZE/4) +tx];
	    else
		As(tx, (ty + BLOCK_SIZE/4)) = 0.0;// A[ a+ wA *(ty+ BLOCK_SIZE/4) +tx];

	    if( bflag  +  BLOCK_SIZE/2< N  && aflag <N  )
		As(tx, (ty + BLOCK_SIZE/2)) = A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];
	    else 
		As(tx, (ty + BLOCK_SIZE/2)) = 0.0;// A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];

	    if( bflag +  BLOCK_SIZE/4*3< N && aflag <N  )
		As(tx, (ty + BLOCK_SIZE/4*3)) = A[ a+ wA *(ty+ BLOCK_SIZE/4*3) +tx];
	    else
		As(tx, (ty + BLOCK_SIZE/4*3)) = 0.0; //A[ a+ wA *(ty+ BLOCK_SIZE/4*3) +tx];


	    aflag = step_no * BLOCK_SIZE + ty;     
	    bflag = bx * BLOCK_SIZE +tx ;
	    if ( aflag < N && bflag  < N )  Bs(tx, ty) = B[b + wB*ty +tx];
	    else Bs(tx, ty) = 0. ;

	    if( aflag  +  BLOCK_SIZE/4 < N && bflag  < N  ) 
		Bs(tx, (ty + BLOCK_SIZE/4)) = B[ b + wB *(ty+ BLOCK_SIZE/4) +tx];
	    else
		Bs(tx, (ty + BLOCK_SIZE/4)) = 0.0;// A[ a+ wA *(ty+ BLOCK_SIZE/4) +tx];

	    if( aflag  +  BLOCK_SIZE/2< N  && bflag <N  )
		Bs(tx, (ty + BLOCK_SIZE/2)) = B[ b+ wB *(ty+ BLOCK_SIZE/2) +tx];
	    else 
		Bs(tx, (ty + BLOCK_SIZE/2)) = 0.0;// A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];

	    if( aflag +  BLOCK_SIZE/4*3< N && bflag <N  )
		Bs(tx, (ty + BLOCK_SIZE/4*3)) = B[ b+ wB *(ty+ BLOCK_SIZE/4*3) +tx];
	    else
		Bs(tx, (ty + BLOCK_SIZE/4*3)) = 0.0; //A[ a+ wA *(ty+ BLOCK_SIZE/4*3) +tx];


	    __syncthreads();
#pragma unroll
	    for (int k = 0 ; k < BLOCK_SIZE; k++)
	    {
		Csub[0] += As(k,ty) * Bs(tx, k);
		Csub[1] += As(k, ty + BLOCK_SIZE/4) * Bs(tx, k) ;
		Csub[2] += As(k, ty + BLOCK_SIZE/2) * Bs(tx, k) ;
		Csub[3] += As(k, ty + BLOCK_SIZE/4*3) * Bs(tx, k) ;
	    }
	    __syncthreads();

	}
	int c = wB* BLOCK_SIZE * by + BLOCK_SIZE *bx;
	if( bx * BLOCK_SIZE + tx < N && by * BLOCK_SIZE + ty < N){
	    C[c+ wB *ty + tx] = Csub[0];
	}
	if( bx * BLOCK_SIZE + tx < N && by * BLOCK_SIZE + ty + BLOCK_SIZE/4 < N){
	    C[c+ wB *(ty+ BLOCK_SIZE/4) + tx] = Csub[1];
	}
	if( bx * BLOCK_SIZE + tx < N && by * BLOCK_SIZE + ty + BLOCK_SIZE /2  <  N){
	    C[c+ wB *(ty+ BLOCK_SIZE/2) + tx] = Csub[2];
	}
	if( bx * BLOCK_SIZE + tx < N && by * BLOCK_SIZE + ty + BLOCK_SIZE/4*3  <  N){
	    C[c+ wB *(ty+ BLOCK_SIZE/4*3) + tx] = Csub[3];
	}
    }

}





/*	
	    if ( a - aBegin + tx < N && by * BLOCK_SIZE + ty < N ){
		As(tx, ty) = A[a + wA*ty +tx];
		if( a - aBegin + tx  +  BLOCK_SIZE/4< N){

		    As(tx, (ty + BLOCK_SIZE/4)) = A[ a+ wA *(ty+ BLOCK_SIZE/4) +tx];

		    if(a - aBegin + tx   +  BLOCK_SIZE/2< N  ){
			As(tx, (ty + BLOCK_SIZE/2)) = A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];

			if(a - aBegin + tx +   BLOCK_SIZE/4*3< N  ){

			    As(tx, (ty + BLOCK_SIZE/4*3)) = A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];

			}
			else{

			    As(tx, (ty + BLOCK_SIZE/4*3)) = 0.0; //A[ a+ wA *(ty+ BLOCK_SIZE/4*3) +tx];
			}
		    }
		    else{
			As(tx, (ty + BLOCK_SIZE/2)) = 0.0; //A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];
			As(tx, (ty + BLOCK_SIZE/4*3)) = 0.0 ;//  A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];
		    }
		}
		else{
		    As(tx, (ty + BLOCK_SIZE/4)) =0.0;// A[ a+ wA *(ty+ BLOCK_SIZE/4) +tx];
		    As(tx, (ty + BLOCK_SIZE/2)) = 0.0; //A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];
		    As(tx, (ty + BLOCK_SIZE/4*3)) = 0.0 ;//  A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];
		}
	    }
	    else{
		As(tx, ty) = 0.0 ;//A[a + wA*ty +tx];
		As(tx, (ty + BLOCK_SIZE/4)) =0.0;// A[ a+ wA *(ty+ BLOCK_SIZE/4) +tx];
		As(tx, (ty + BLOCK_SIZE/2)) = 0.0; //A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];
		As(tx, (ty + BLOCK_SIZE/4*3)) = 0.0;//  A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];
	    }
*/
/*
	    // The BS assign
	    if (  step_no * BLOCK_SIZE + ty < N && bx * BLOCK_SIZE +tx < N ){
		Bs(tx, ty) = B[b + wB*ty +tx];

		if( step_no * BLOCK_SIZE + ty +  BLOCK_SIZE/4< N){

		    Bs(tx, (ty + BLOCK_SIZE/4)) = B[ b + wB *(ty+ BLOCK_SIZE/4) +tx];

		    if( step_no * BLOCK_SIZE + ty +  BLOCK_SIZE/2< N  ){
			Bs(tx, (ty + BLOCK_SIZE/2)) = B[ b + wB *(ty+ BLOCK_SIZE/2) +tx];

			if( step_no * BLOCK_SIZE + ty +  BLOCK_SIZE/4*3< N  ){

			    Bs(tx, (ty + BLOCK_SIZE/4*3)) = B[ b + wB *(ty+ BLOCK_SIZE/4*3) +tx];

			}
			else{

			    Bs(tx, (ty + BLOCK_SIZE/4*3)) = 0.0; //A[ a+ wA *(ty+ BLOCK_SIZE/4*3) +tx];
			}
		    }
		    else{
			Bs(tx, (ty + BLOCK_SIZE/2)) = 0.0; //A[ a+ wA *(ty+ BLOCK_SIZE/2) +tx];
			Bs(tx, (ty + BLOCK_SIZE/4*3)) = 0.0;
		    }
		}
		else{
		    Bs(tx, (ty + BLOCK_SIZE/4)) =0.0;// A[ a+ wA *(ty+ BLOCK_SIZE/4) +tx];
		    Bs(tx, (ty + BLOCK_SIZE/2)) = 0.0;//B[ b + wB *(ty+ BLOCK_SIZE/2) +tx];
		    Bs(tx, (ty + BLOCK_SIZE/4*3)) = 0.0;
		}

	    }
	    else{
		Bs(tx, ty) = 0.0 ;//A[a + wA*ty +tx];
		Bs(tx, (ty + BLOCK_SIZE/4)) = 0.0;// B[ b + wB *(ty+ BLOCK_SIZE/4) +tx];
		Bs(tx, (ty + BLOCK_SIZE/2)) = 0.0;//B[ b + wB *(ty+ BLOCK_SIZE/2) +tx];
		Bs(tx, (ty + BLOCK_SIZE/4*3)) = 0.0;//B[ b + wB *(ty+ BLOCK_SIZE/4*3) +tx];
	    }
*/

