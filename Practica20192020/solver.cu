#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "cuda_runtime.h"
#include "solver.h"

using namespace std;

/**
* Kernel del calculo de la solvation. Se debe anadir los parametros 
*/
__global__ void escalculation (int atoms_r, int atoms_l, int nlig, float *rec_x_d, float *rec_y_d, float *rec_z_d, float *lig_x_d, float *lig_y_d, float *lig_z_d, float *ql_d,float *qr_d, float *energy_d, int nconformations){
	
	int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float dist, total_elec = 0, miatomo[3], elecTerm;//
    int totalAtomLig = nconformations * nlig;
    //Para depuración, borrar para producción/////////////////////////////////
	if(row==7 && col==7){
		printf("atoms_r:%d,atoms_l:%d\n",atoms_r,atoms_l);
		printf("gridDim(x*y):%d*%d bloques\n",gridDim.x,gridDim.y);
		printf("Hilos bloque(x*y):%d*%d-total %d hilos\n",blockDim.x,blockDim.y,blockDim.x*blockDim.y);
		printf("Hola soy el hilo col:%d\n",col);
		printf("Hola soy el hilo row:%d\n",row);
	}
    if (col < atoms_l && row < atoms_r){
        for (int k = 0; k < totalAtomLig; k += nlig) {
            miatomo[0] = *(lig_x_d + k + col);
            miatomo[1] = *(lig_y_d + k + col);
            miatomo[2] = *(lig_z_d + k + col);
            dist = calculaDistancia(rec_x_d[row], rec_y_d[row], rec_z_d[row], miatomo[0], miatomo[1], miatomo[2]);              
            atomicAdd(&energy_d[k / nlig], (ql_d[col] * qr_d[row]) / dist);//explicado en cudabyexample pagina179         
        }
  }

}
//***Explicación de la sentencia atómica atomicAdd(&energy_d[k / nlig], (ql_d[col] * qr_d[row]) / dist); 
//1º lee &energy_d[k / nlig]
//2º suma a lo leido esto:(ql_d[col] * qr_d[row]) / dist)
//3º escribe el resultado en lee &energy_d[k / nlig]
//Evita condiciones de carrera ya que solo un hilo puede accede a escribir a la vez, en el libro indica
//que está garantizado por hardware
/**
* Funcion para manejar el lanzamiento de CUDA 
*/
void forces_GPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){

	cudaError_t cudaStatus; //variable para recoger estados de cuda
	//seleccionamos device
	cudaSetDevice(0); //0 - Tesla K40 vs 1 - Tesla K230
	//creamos memoria para los vectores para GPU _d (device)
	float *rec_x_d, *rec_y_d, *rec_z_d, *qr_d, *lig_x_d, *lig_y_d, *lig_z_d, *ql_d, *energy_d;
	//////////////reservamos memoria para GPU///////////////////////////////////////////////////////////
	int memsizeatom_r   = atoms_r * sizeof(float);
	int memsizeatom_l   = atoms_l * sizeof(float);
    int memsizeligandos = nlig * nconformations * sizeof(float);//nos comentó en una video.
    int memsizeql       = nlig * sizeof(float);
    int memsizeenergy   = nconformations * sizeof(float);
 
	cudaStatus = cudaMalloc((void**)&energy_d, memsizeenergy);
	cudaStatus = cudaMalloc((void**)&qr_d, memsizeatom_r);
	cudaStatus = cudaMalloc((void**)&ql_d, memsizeql);
    cudaStatus = cudaMalloc((void**)&rec_x_d, memsizeatom_r);
	cudaStatus = cudaMalloc((void**)&lig_x_d, memsizeligandos);
    cudaStatus = cudaMalloc((void**)&rec_y_d, memsizeatom_r);
	cudaStatus = cudaMalloc((void**)&lig_y_d, memsizeligandos);
    cudaStatus = cudaMalloc((void**)&rec_z_d, memsizeatom_r);
	cudaStatus = cudaMalloc((void**)&lig_z_d, memsizeligandos);
 	////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////Pass data to the device///////////////////////////////////////////////////////
    cudaStatus = cudaMemcpy(energy_d, energy, memsizeenergy, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(qr_d, qr, memsizeatom_r, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(ql_d, ql, memsizeql, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(rec_x_d, rec_x, memsizeatom_r, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(lig_x_d, lig_x, memsizeligandos, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(rec_y_d, rec_y, memsizeatom_r, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(lig_y_d, lig_y, memsizeligandos, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(rec_z_d, rec_z, memsizeatom_r, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(lig_z_d, lig_z, memsizeligandos, cudaMemcpyHostToDevice);   
    ////////////////////////////////////////////////////////////////////////////////////////////////////

	////////////////pasamos datos de host to device/////////////////////////////////////////////////////
	cudaStatus = cudaMemcpy(energy_d, energy, memsizeenergy, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(qr_d, qr, memsizeatom_r, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(ql_d, ql, memsizeql, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(rec_x_d, rec_x, memsizeatom_r, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(lig_x_d, lig_x, memsizeligandos, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(rec_y_d, rec_y, memsizeatom_r, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(lig_y_d, lig_y, memsizeligandos, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(rec_z_d, rec_z, memsizeatom_r, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(lig_z_d, lig_z, memsizeligandos, cudaMemcpyHostToDevice); 
	////////////////////////////////////////////////////////////////////////////////////////////////////
	
 	/////////////////////////////Definir numero de hilos y bloques//////////////////////////////////////
	int numthreads_x = 4;
	int numthreads_y = 128;
	int numblocks_x = (atoms_l+numthreads_x-1)/numthreads_x;//este método lo explica en el libro cuda by example
	int numblocks_y = (atoms_r+numthreads_y-1)/numthreads_y;//página 65
	//
	dim3 block(numblocks_x,numblocks_y);
    dim3 thread(numthreads_x,numthreads_y);
	//llamada al kernel
	escalculation <<<block,thread>>> (atoms_r, atoms_l, nlig, rec_x_d, rec_y_d, rec_z_d, lig_x_d, lig_y_d, lig_z_d, ql_d, qr_d, energy_d, nconformations);
	//control de errores kernel
	cudaDeviceSynchronize();
	
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Error en el kernel %d\n", cudaStatus); 

	//Traemos info al host
    cudaStatus = cudaMemcpy(energy, energy_d, memsizeenergy, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(qr, qr_d, memsizeatom_r, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(rec_x, rec_x_d, memsizeatom_r, cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(lig_x, lig_x_d, memsizeligandos, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(rec_y, rec_y_d, memsizeatom_r, cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(lig_y, lig_y_d, memsizeligandos, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(rec_z, rec_z_d, memsizeatom_r, cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(lig_z, lig_z_d, memsizeligandos, cudaMemcpyDeviceToHost);

	//para comprobar que la ultima conformacion tiene el mismo resultado que la primera
	printf("Termino electrostatico de conformacion %d es: %f\n", nconformations-1, energy[nconformations-1]); 

	printf("Termino electrostatico %f\n", energy[0]);
	//Liberamos memoria en GPU
	cudaFree(energy_d);
    cudaFree(qr_d);
    cudaFree(ql_d);
	cudaFree(rec_x_d);
	cudaFree(lig_x_d);
    cudaFree(rec_y_d);
	cudaFree(lig_y_d);
    cudaFree(rec_z_d);
    cudaFree(lig_z_d);
}

/**
* Distancia euclidea compartida por funcion CUDA y CPU secuencial
*/
__device__ __host__ extern float calculaDistancia (float rx, float ry, float rz, float lx, float ly, float lz) {

  float difx = rx - lx;
  float dify = ry - ly;
  float difz = rz - lz;
  float mod2x=difx*difx;
  float mod2y=dify*dify;
  float mod2z=difz*difz;
  difx=mod2x+mod2y+mod2z;
  return sqrtf(difx);
}




/**
 * Funcion que implementa el termino electrostático en CPU
 */
void forces_CPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){

	double dist, total_elec = 0, miatomo[3], elecTerm;
  int totalAtomLig = nconformations * nlig;
    printf("Atomos atoms_l:%d\n",atoms_l);
	printf("Atomos atoms_r:%d\n",atoms_r);
	for (int k=0; k < totalAtomLig; k+=nlig){
	  for(int i=0;i<atoms_l;i++){					
			miatomo[0] = *(lig_x + k + i);
			miatomo[1] = *(lig_y + k + i);
			miatomo[2] = *(lig_z + k + i);

			for(int j=0;j<atoms_r;j++){				
				elecTerm = 0;
        dist=calculaDistancia (rec_x[j], rec_y[j], rec_z[j], miatomo[0], miatomo[1], miatomo[2]);
//				printf ("La distancia es %lf\n", dist);
        elecTerm = (ql[i]* qr[j]) / dist;
				total_elec += elecTerm;
//        printf ("La carga es %lf\n", total_elec);
			}
		}
		
		energy[k/nlig] = total_elec;
		total_elec = 0;
  }
	printf("Termino electrostatico %f\n", energy[0]);
}


extern void solver_AU(int mode, int atoms_r, int atoms_l,  int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql, float *qr, float *energy_desolv, int nconformaciones) {

	double elapsed_i, elapsed_o;
	
	switch (mode) {
		case 0://Sequential execution
			printf("\* CALCULO ELECTROSTATICO EN CPU *\n");
			printf("**************************************\n");			
			printf("Conformations: %d\t Mode: %d, CPU\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_CPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("CPU Processing time: %f (seg)\n", elapsed_o);
			break;
		case 1: //OpenMP execution
			printf("\* CALCULO ELECTROSTATICO EN OPENMP *\n");
			printf("**************************************\n");			
			printf("**************************************\n");			
			printf("Conformations: %d\t Mode: %d, CMP\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_OMP_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("OpenMP Processing time: %f (seg)\n", elapsed_o);
			break;
		case 2: //CUDA exeuction
			printf("\* CALCULO ELECTROSTATICO EN CUDA *\n");
      printf("**************************************\n");
      printf("Conformaciones: %d\t Mode: %d, GPU\n",nconformaciones,mode);
			elapsed_i = wtime();
			forces_GPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("GPU Processing time: %f (seg)\n", elapsed_o);			
			break; 	
	  	default:
 	    	printf("Wrong mode type: %d.  Use -h for help.\n", mode);
			exit (-1);	
	} 		
}
