#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "solver.h"

/**
* Funcion que implementa la solvatacion en openmp
*/
extern void forces_OMP_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){

  //Verifico el número de hilos que dispongo
  printf("El max de hilos disponibles es esta maquina es de %d hilos\n",omp_get_max_threads());
  printf("==========================================================\n");
  //Declaración de variables
  float dist, total_elec = 0, miatomo[3], elecTerm;
  int totalAtomLig = nconformations * nlig;
  //==============================Configuración openmp==================================
  omp_set_num_threads(omp_get_max_threads());//establezco al máximo de hilos disponibles.
  //omp_set_num_threads(4);//establezco al máximo de hilos disponibles.
  //#pragma omp parallel for private(dist,elecTerm,miatomo) reduction(+:total_elec)//opción 1.
  #pragma omp parallel for private(dist,elecTerm,miatomo,total_elec)
  //====================================================================================
  for (int k=0; k < totalAtomLig; k+=nlig) {
    //Análisis de distribución de hilos
    //printf("Bucle 0: Thread %d hace la iter %d de k de un total de %d\n",omp_get_thread_num(),k,totalAtomLig);
    for(int i=0;i<atoms_l;i++){
    //printf("Bucle-1: Th%d hace la iter %d de i de un total de %d\n",omp_get_thread_num(),i,atoms_l);
      miatomo[0] = *(lig_x + k + i);
      miatomo[1] = *(lig_y + k + i);
      miatomo[2] = *(lig_z + k + i);            
        for(int j=0;j<atoms_r;j++){
          elecTerm = 0;
          dist=calculaDistancia (rec_x[j], rec_y[j], rec_z[j], miatomo[0], miatomo[1], miatomo[2]);
          elecTerm = (ql[i]* qr[j]) / dist;
          total_elec += elecTerm;
        }
     }
     energy[k/nlig] = total_elec;
     total_elec = 0;
  }
  printf("Termino electrostatico %f\n", energy[0]);
}


