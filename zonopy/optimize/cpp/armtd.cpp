#include "IpTNLP.hpp"
#include "IpIpoptApplication.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "armtd.h"

//#include <ctime>


/*
self.qpos
self.qvel
qgoal
n_timestep = 100
n_links
n_obs

<list>
self.A
self.b
c
n_ids
G
expMat[:,torch.argsort(self.id)]


*/



armtd_NLP::armtd_NLP()
{
    qpos = nullptr;
    qvel = nullptr;
    qgoal = nullptr;

    A = nullptr;
    b = nullptr;

    g_k = nullptr;
    x_0 = nullptr;
}

// destructor
armtd_NLP::~armtd_NLP()
{
    delete[] constraints_FO;
    delete[] jacobian_FO;
}



bool armtd_NLP::set_parameters(
    TYPE* qpos,
    TYPE* qvel, 
    TYPE* qgoal, 
    int n_joints, 
    int n_obs, 
    TYPE* A,
    TYPE* b,
    TYPE* c, 
    int* n_ids,
    int* n_gens,
    TYPE* G, 
    int* expMat
)
{
    // pybind
}

// [TNLP_get_nlp_info]
// 
bool armtd_NLP::get_nlp_info(
    Index&          n,
    Index&          m,
    Index&          nnz_jac_g,
    Index&          nnz_h_lag,
    IndexStyleEnum& index_style
)
{
    // number of decision variables (dim. of trajectory parameter)
    n = n_joints;

    // number of inequality constraints
    m = n_joints*n_timesteps*n_obs;
    nnz_jac_g = m * n;

    // use the C style indexing (0-based)
    index_style = TNLP::C_STYLE;

    return true;
}
// [TNLP_get_nlp_info]

// [TNLP_get_bounds_info]
// returns bounds on variables and constraints
bool armtd_NLP::get_bounds_info(
    Index   n,
    Number* x_l,
    Number* x_u,
    Index   m,
    Number* g_l,
    Number* g_u
)
{
    // the n and m fed into IPOPT from get_nlp_info will return.
    // If desired, we could assert to make sure they are what we think they are.
    if(n != n_joints){
        printf("MyProg:ConvertString *** Error wrong value of n in get_bounds_info!");
    }

    if(m != n_joints*n_timesteps*n_obs){
        printf("MyProg:ConvertString *** Error wrong value of m in get_bounds_info!");
    }

    // bounds on variable
    for( Index i = 0; i < n; i++ ) {
        x_l[i] = -1.0;
        x_u[i] = 1.0;
    }

    // bounds on  constraints
    for( Index i = 0; i < m; i++ ) {
        g_l[i] = -1e19;
        g_u[i] = 0;
    }    
    return true;
}
// [TNLP_get_bounds_info]

// [TNLP_get_starting_point]
// returns the initial point for the problem
bool armtd_NLP::get_starting_point(
    Index   n,
    bool    init_x,
    Number* x,
    bool    init_z,
    Number* z_L,
    Number* z_U,
    Index   m,
    bool    init_lambda,
    Number* lambda
)
{
    // Here, we only give initial guess for decision variable
    // you can provide starting values for the dual variables as well
    if(init_x == false || init_z == true || init_lambda == true){
        printf("MyProg:ConvertString *** Error wrong value of init in get_starting_point!");
    }

    if(n != n_joints){
        printf("MyProg:ConvertString *** Error wrong value of n in get_starting_point!");
    }

    // initialize to a random point
    std::srand(std::time(NULL));
    for( Index i = 0; i < n; i++ ) {
        //x[i] = 2.0 * ((TYPE)std::rand() / RAND_MAX) - 1.0;
        x[i] = 0;
    }

    return true;
}
// [TNLP_get_starting_point]


// [TNLP_eval_f]
// returns the value of the objective function
bool armtd_NLP::eval_f(
    Index         n,
    const Number* x,
    bool          new_x,
    Number&       obj_value
)
{
    if(n != n_joints){
        printf("MyProg:ConvertString *** Error wrong value of n in eval_f!");
    }

    // q_plan = q + q_dot*P.t_plan + 0.5*k*P.t_plan^2;
    // obj_value = sum((q_plan - q_des).^2);
    obj_value = 0; 
    for(Index i = 0; i < n_joints; i++){
        // TODO: QC: here modified
        // TYPE entry = q0[i] + v0[i] * t_plan + x[i] * t_plan * t_plan / 2 - q_des[i];
        TYPE entry = qpos[i] + qvel[i] * t_plan * 1.5 + g_k[i] * x[i] * t_plan * t_plan - q_des[i];
        obj_value += entry * entry;
    }

    return true;
}
// [TNLP_eval_f]


// [TNLP_eval_grad_f]
// return the gradient of the objective function grad_{x} f(x)
bool armtd_NLP::eval_grad_f(
    Index         n,
    const Number* x,
    bool          new_x,
    Number*       grad_f
)
{
    if(n != n_joints){
        printf("MyProg:ConvertString *** Error wrong value of n in eval_grad_f!");
    }

    for(Index i = 0; i < n_joints; i++){
        // TODO: QC: modified
        // TYPE entry = q0[i] + v0[i] * t_plan + x[i] * t_plan * t_plan / 2 - q_des[i];
        TYPE entry = qpos[i] + qvel[i] * t_plan * 1.5 + g_k[i] * x[i] * t_plan * t_plan - q_des[i];
        grad_f[i] = 2 * g_k[i] * t_plan * t_plan * entry;
    }
    return true;
}
// [TNLP_eval_grad_f]

// [TNLP_eval_g]
// return the value of the constraints: g(x)
bool armtd_NLP::eval_g(
    Index         n,
    const Number* x,
    bool          new_x,
    Index         m,
    Number*       g
)
{
    if(n != n_joints){
        printf("MyProg:ConvertString *** Error wrong value of n in eval_g!");
    }
    if(m != n_joints*n_timesteps*n_obs){
        printf("MyProg:ConvertString *** Error wrong value of m in eval_g!");
    }

    // detect if the result has been computed at current point x
    bool compute_new_constraints = false;

    for (Index i = 0; i < n_joints; i++) {
        if (current_x[i] != x[i]) {
            compute_new_constraints = true;
            break;
        }
    }

    if(compute_new_constraints){
        std::memcpy(current_x, x, n * sizeof(Number));
        eval_FO_constraints(c,G,expMat,n_ids,A,b,n_obs,constraints_FO,x);

        /*
        TYPE k[n_joints];
        for (int i = 0; i < n; i++) {
            k[i] = x[i] * g_k[i];
        } 
        compute_max_min_states(k);
        */
    }

    std::memcpy(g, constraints_FO, n_joints * n_timesteps * n_obs * sizeof(TYPE));

    /*
    Index offset = n_joints * n_timesteps * n_obs;

    offset += n_joints * n_timesteps;
    for(Index i = offset; i < offset + n_obs; i++) {
        g[i] = q_min[i - offset];
    }
    offset += n_joints;
    for(Index i = offset; i < offset + n_joints; i++) {
        g[i] = q_max[i - offset];
    }
    offset += n_joints;
    for(Index i = offset; i < offset + n_joints; i++) {
        g[i] = v_min[i - offset];
    }
    offset += n_joints;
    for(Index i = offset; i < offset + n_joints; i++) {
        g[i] = v_max[i - offset];
    }
    */
    return true;
}
// [TNLP_eval_g]

// [TNLP_eval_jac_g]
// return the structure or values of the Jacobian
bool armtd_NLP::eval_jac_g(
   Index         n,
   const Number* x,
   bool          new_x,
   Index         m,
   Index         nele_jac,
   Index*        iRow,
   Index*        jCol,
   Number*       values
)
{
    if(n != n_joints){
        printf("MyProg:ConvertString *** Error wrong value of n in eval_g!");
    }
    if(m != n_joints*n_timesteps*n_obs){
        printf("MyProg:ConvertString *** Error wrong value of m in eval_g!");
    }
        
    if( values == NULL ) {
       // return the structure of the Jacobian
       // this particular Jacobian is dense
        for(Index i = 0; i < m; i++){
            for(Index j = 0; j < n; j++){
                iRow[i * n + j] = i;
                jCol[i * n + j] = j;
            }
        }
    }
    else {
        // detect if the result has been computed at current point x
        bool compute_new_constraints = false;

        for (Index i = 0; i < n; i++) {
            if (current_x[i] != x[i]) {
                compute_new_constraints = true;
                break;
            }
        }

        if(compute_new_constraints){
            std::memcpy(current_x, x, n * sizeof(Number));
            
            grad_FO_constraints(c,G,expMat,n_ids,A,b,n_obs,jacobian_FO)
            /*
            TYPE k[n_joints];
            for (int i = 0; i < n; i++) {
                k[i] = x[i] * g_k[i];
            } 
            
            compute_max_min_states(k);
            */
        }

        // return the values of the Jacobian of the constraints
        std::memcpy(values, jacobian_FO, n_joints * n_timesteps * n_obs * n * sizeof(TYPE));
        /*
        Index offset = n_joints * n_timesteps * n_obs;
        for(Index i = offset; i < offset + n; i++) {
            for(Index j = 0; j < n; j++){
                if(i - offset == j){
                    values[i * n + j] = grad_q_min[j];
                }  
                else{
                    values[i * n + j] = 0;
                }
            }
        }
        offset += n;
        for(Index i = offset; i < offset + n; i++) {
            for(Index j = 0; j < n; j++){
                if(i - offset == j){
                    values[i * n + j] = grad_q_max[j];
                }  
                else{
                    values[i * n + j] = 0;
                }
            }
        }
        offset += n;
        for(Index i = offset; i < offset + n; i++) {
            for(Index j = 0; j < n; j++){
                if(i - offset == j){
                    values[i * n + j] = grad_v_min[j];
                }  
                else{
                    values[i * n + j] = 0;
                }
            }
        }
        offset += n;
        for(Index i = offset; i < offset + n; i++) {
            for(Index j = 0; j < n; j++){
                if(i - offset == j){
                    values[i * n + j] = grad_v_max[j];
                }  
                else{
                    values[i * n + j] = 0;
                }
            }
        }
        */
    } 
    return true;
}
// [TNLP_eval_jac_g]


// [TNLP_eval_h]
//return the structure or values of the Hessian
bool armtd_NLP::eval_h(
   Index         n,
   const Number* x,
   bool          new_x,
   Number        obj_factor,
   Index         m,
   const Number* lambda,
   bool          new_lambda,
   Index         nele_hess,
   Index*        iRow,
   Index*        jCol,
   Number*       values
)
{
    return false;
}
// [TNLP_eval_h]



// [TNLP_finalize_solution]
void armtd_NLP::finalize_solution(
   SolverReturn               status,
   Index                      n,
   const Number*              x,
   const Number*              z_L,
   const Number*              z_U,
   Index                      m,
   const Number*              g,
   const Number*              lambda,
   Number                     obj_value,
   const IpoptData*           ip_data,
   IpoptCalculatedQuantities* ip_cq
)
{
   // here is where we would store the solution to variables, or write to a file, etc
   // so we could use the solution.

   // store the solution
   for( Index i = 0; i < n; i++ ) {
      solution[i] = (TYPE)x[i];
   }
}
// [TNLP_finalize_solution]

