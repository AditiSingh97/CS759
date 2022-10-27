// omp_get_num_threads.cpp
// compile with: /openmp or -fopenmp
#include <cstdio>
#include <iostream>
#include <omp.h>

int factorial(int n){
	if((n == 0) || (n == 1)){
		return 1;
	}
	return n * factorial(n-1);
}

int main() {
	omp_set_num_threads(4);
	// NB: run-time action, sets OpenMP behavior
#pragma omp parallel
#pragma omp master
	{
		std::printf("Number of threads: %d\n", omp_get_num_threads());
	}
#pragma omp parallel
	{
		std::printf("I am thread No. %d\n", omp_get_thread_num());
	}
#pragma omp parallel
	{
		int num1 = omp_get_thread_num()*2+1;
		int num2 = omp_get_thread_num()*2+2;
		int fact1 = factorial(num1);
		int fact2 = factorial(num2);

		printf("%d!=%d\n",num1, fact1);
		printf("%d!=%d\n",num2, fact2);
	}

	return 0;
}
