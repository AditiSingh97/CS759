#include<iostream>
int main(int argc, char **argv){
    if(argc != 2){
        exit(-1);
    }

    int N  = atoi(argv[1]);

    for(int i = 0; i <= N; i++){
        if(i == N){
            printf("%d\n", i);
        }
        else{
            printf("%d ", i);
        }
    }

    for(int i = N; i >= 0; i--){
        if(i == 0){
            std::cout << i << std::endl;
        }
        else{
            std::cout << i << " ";
        }
    }
    return 0;
}
