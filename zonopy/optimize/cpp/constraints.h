#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H


void eval_FO_constraints(c*,G*,expMat*,n_ids*,A*,b*,int n_obs,constraints_FO*);

void grad_FO_constraints(c*,G*,expMat*,n_ids*,A*,b*,int n_obs,jacobian_FO*);

#endif