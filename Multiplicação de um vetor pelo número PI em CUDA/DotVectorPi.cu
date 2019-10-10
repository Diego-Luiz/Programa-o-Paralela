#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define max 1024 //Tamanho do Vetor
#define pi 3.1415 //numero PI com 4 casas decimais

/*Função que será executada pelos blocos da GPU e suas threads*/
__global__ void mpi(double *a, double *b){
      // id = id do bloco * dimensão do bloco * thread do block
      int id= blockIdx.x * blockDim.x + threadIdx.x;
    
      /*
      Verificação necessária para evitar acessos indevidos a memória
      */
      if(id< max) 
        b[id]= a[id]*pi; //realizando a operação de multiplicação
}

int main(){
    /*Variável para armazenar o tempo final gasto no processamento*/
    double time=0.0; 
    
    /*
      --> Como as variáveis estão sendo utilizadas:
      - h_a : ponteiro do tipo double que armazena o endereço do
      vetor na CPU e que será preenchido com seu índice de 
      posição + 1 posição
      - h_b : ponteiro do tipo double que armazena o endereço do
      vetor na CPU onde será guardado o resultado da operação de
      multiplicação do vetor h_a
      - d_a : ponteiro do tipo double que armazena o endereço do
      vetor na GPU e que receberá o vetor h_a pela função
      cudaMemcpy()
      - d_b : ponteiro do tipo double que armazena o endereço do
      vetor na GPU e que receberá o resultado da operação da
      multiplicação do vetor d_a
    */
    
    /*
        Variáveis do tipo double e ponteiro, que são usados pela CPU e
        que armazenam o endereço dos vetores que serão alocados 
    */
    
    double *h_a, *h_b;
    
    /*
        Alocando dinamicamente um vetor na memória de tamanho 
        max * tamanho de double
    */
    h_a= (double*) malloc(max*sizeof(double));
    
    /*
        Alocando dinamicamente um vetor na memória de tamanho 
        max * tamanho de double
    */
    h_b= (double*) malloc(max*sizeof(double));
    
    /*
        Laco que inicia cada posição da memória com seu
        respectivo índice + 1 unidade
    */
    for(int i=0;i<max;i++){
        h_a[i]= i+1;
    }
    /*
        Variáveis do tipo double e ponteiro que guardarão os
        endereços dos vetores na GPU 
    */
    double *d_a, *d_b;
    
    /*
        Alocando um vetor na memoria da GPU de tamanho 
        max * tamanho de double
    */
    cudaMalloc(&d_a,max*sizeof(double));
    /*
        Alocando um vetor na memoria da GPU de tamanho 
        max * tamanho de double
    */
    cudaMalloc(&d_b,max*sizeof(double));
    
    /*
      Função que copia o conteúdo do vetor h_a (CPU) para o vetor 
      d_a (CPU)
      Parâmetros: (destino, origem, tamanho do elemento a ser
      copiado, tipo de cópia)
    */
    cudaMemcpy(d_a,h_a,max*sizeof(double),cudaMemcpyHostToDevice);
    
    //---------------------------------------------------------------
    /*Variável do tipo clock_t, que inicia a contagem do tempo*/
    clock_t begin=clock(); 
    
    /*
      Invocando método responsável por realizar a operação
      Parâmetros: nome_função<<<numero de blocos, qtd
      threads>>>(argumento 1, argumento 2);
    */
    mpi<<<1024,1>>>(d_a,d_b);
    
    
    /*Variável do tipo clock_t, que termina a contagem do tempo*/
    clock_t end=clock(); 
    //---------------------------------------------------------------
    
    /*
      Função que copia o conteúdo do vetor d_b (GPU) para o vetor h_b
      (CPU)
      Parâmetros: (destino, origem, tamanho do elemento a ser
      copiado, tipo de copia)
    */
    cudaMemcpy(h_b, d_b, max * sizeof(double), cudaMemcpyDeviceToHost);
    
    /*Calculando o tempo em segundos*/
    time+= (double)(end - begin) / CLOCKS_PER_SEC;
    
    /*Exibindo o resultado final*/
    printf("Tempo gasto: %f segundos", time); 
    
    /*Liberando a memória da GPU*/
    cudaFree(d_a);
    cudaFree(d_b);
    
    /*Liberando a memória da CPU*/
    free(h_a);
    free(h_b);
    
    
    return 0;
}