#ifndef CONSTRAINTS_CPP
#define CONSTRAINTS_CPP
#include "constraints.h"


inline void eval_FO_constraints(TYPE* c,TYPE* G,expMat, TYPE* A, TYPE* b, int* n_gens, int* n_ids, Index n_joints, Index n_timesteps, Index n_obs, TYPE* val_slc, TYPE* constraints_FO){

    std::memcpy(slc_c, c, n_joints*n_obs*dimension * sizeof(TYPE));

    for( Index j = 0; j < n_joints; j++ ) {
        for ( Index o = 0; o < n_obs; o++) {
            // mul = ones(n_gens)
            TYPE mul[n_gens] = {0.0};
            for ( Index g = 0; g<n_gens[j]; g++) {            
                mul[g] = 1.0;          
                for ( Index k = 0; k < j; k++){
                    //[((j*n_obs+o)*n_gens[j]+g)*j+k]  
                    mul[g] *= POW(val_slc[k],expMat[j,o][g,k]);
                }   
            }
            // slc_c = zeros(dimension)
            //TYPE slc_c[n_gens] = {0.0};
            for ( Index d = 0; d<dimension; d++){
                // slc_c[j,o,d] = c[j,o][d] 
                for ( Index g=0; g<n_gens; g++){
                    slc_c[j,o,d] += G[j,o][g,d]*mul[g]                
                }
            }
            // A,b

        }
    }
}





inline void grad_FO_constraints(c*,G*,expMat*,A*,b*, n_ids*,Index n_joints, Index n_timesteps, Index n_obs, val_slc*,constraints_FO*){
    for( Index j = 0; j < n_joints; j++ ) {
        for ( Index o = 0; o < n_obs; o++) {
            // mul = ones(n_gens)
            // grad_mul = zeros(j,n_gens)
            for ( Index g = 0; g<n_gens; g++) {            
                for ( Index k = 0; k < j; k++){                
                    mul[g] = mul[g]*pow(val_slc[k],expMat[j,o][g,k]);
                    grad_mul[k,g] = expMat[j,o][g,k];
                    for ( Index kk = 0; kk<j; kk++){
                        if (kk == k){
                            exp = expMat[j,o][g,k] -1
                        }
                        else{
                            exp = expMat[j,o][g,k]
                        }
                        grad_mul[kk,g] = grad_mul[kk,g]*pow(val_slc[k],exp)
                    }
                }   
            }



            
            // slc_c = ones(dimension)
            // slc_grad_c = zeros(dimension,n_joints)
            for ( Index d = 0; d<dimension; d++){
                slc_c[j,o,d] = c[j,o][d] 
                for ( Index g=0; g<n_gens; g++){
                    slc_c[j,o,d] += G[j,o][g,d]*mul[g]                
                    for ( Index k=0; k<j; k++){
                        slc_grad_c[j,o,d] += G[j,o][g,d]*grad_mul[k,g]
                    }
                }
            }
            // A,b

        }
    }
}




