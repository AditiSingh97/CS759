//
// Created by singh273 on 11/23/22.
//
#include <cstddef>
#include "optimize.h"

data_t *get_vec_start(vec *v){
    return v->data;
}

void optimize1(vec *v, data_t *dest){
    int length = v->len;
    data_t *d = get_vec_start(v);
    data_t temp = IDENT;
    for (int i = 0; i < length; i++)
        temp = temp OP d[i];
    *dest = temp;
}
void optimize2(vec *v, data_t *dest){
    int length = v->len;
    int limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    int i;
    // reduce 2 elements at a time
    for (i = 0; i < limit; i += 2) {
        x = (x OP d[i]) OP d[i + 1];
    }
    // Finish any remaining elements
    for (; i < length; i++) {
        x = x OP d[i];
    }
    *dest = x;
}
void optimize3(vec *v, data_t *dest){
    int length = v->len;
    int limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    int i;
    /* reduce 2 elements at a time */
    for (i = 0; i < limit; i += 2) {
        x = x OP(d[i] OP d[i + 1]);
    }
    /* Finish any remaining elements */
    for (; i < length; i++) {
        x = x OP d[i];
    }
    *dest = x;
}
void optimize4(vec *v, data_t *dest){
    long length = v->len;
    long limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x0 = IDENT;
    data_t x1 = IDENT;
    long i;
    // reduce 2 elements at a time
    for (i = 0; i < limit; i += 2) {
        x0 = x0 OP d[i];
        x1 = x1 OP d[i + 1];
    }
    // Finish any remaining elements
    for (; i < length; i++) {
        x0 = x0 OP d[i];
    }
    *dest = x0 OP x1;
}
void optimize5(vec *v, data_t *dest){
    int length = v->len;
    data_t *d = get_vec_start(v);
    data_t temp1 = IDENT;
    data_t temp2 = IDENT;
    data_t temp3 = IDENT;
    int i;
    for (i = 0; i < length; i+=3) {
        temp1 = temp1 OP d[i];
        temp2 = temp2 OP d[i+1];
        temp3 = temp3 OP d[i+2];
    }
    for(; i < length; i++){
        temp1 = temp1 OP d[i];
    }
    *dest = (temp1 OP temp2) OP temp3;
}
