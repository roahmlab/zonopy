#ifndef ARMTD_H
#define ARMTD_H

#include <ctime>

#include "IpTNLP.hpp"
#include "IpIpoptApplication.hpp"

#define t_plan 0.5
#define t_total 1.0
#define t_move 0.5
#define n_timestep 100

using namespace Ipopt;



#define IF_USE_DOUBLE true

#if IF_USE_DOUBLE == true
	#define TYPE double
	#define TYPE_MAX DBL_MAX
	#define TYPE_MIN DBL_MIN
	#define MEX_CLASS mxDOUBLE_CLASS
	#define POW pow
#else
	#define TYPE float
	#define TYPE_MAX FLT_MAX
	#define TYPE_MIN FLT_MIN
	#define MEX_CLASS mxSINGLE_CLASS
	#define POW powf
#endif


class armtd_NLP: public TNLP
{
public:
    /** Default constructor */
    armtd_NLP();

    /** Default destructor */
    virtual ~armtd_NLP();

    // [set_parameters]
    bool set_parameters(
        TYPE* qpos
        TYPE* qvel 
        TYPE* qgoal 
        int n_joints 
        int n_obs 
        TYPE* A
        TYPE* b
    );



    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the NLP */
    virtual bool get_nlp_info(
        Index&          n,
        Index&          m,
        Index&          nnz_jac_g,
        Index&          nnz_h_lag,
        IndexStyleEnum& index_style
    );

    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(
        Index   n,
        Number* x_l,
        Number* x_u,
        Index   m,
        Number* g_l,
        Number* g_u
    );

    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(
        Index   n,
        bool    init_x,
        Number* x,
        bool    init_z,
        Number* z_L,
        Number* z_U,
        Index   m,
        bool    init_lambda,
        Number* lambda
    );

    /** Method to return the objective value */
    virtual bool eval_f(
        Index         n,
        const Number* x,
        bool          new_x,
        Number&       obj_value
    );

    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(
        Index         n,
        const Number* x,
        bool          new_x,
        Number*       grad_f
    );

    /** Method to return the constraint residuals */
    virtual bool eval_g(
        Index         n,
        const Number* x,
        bool          new_x,
        Index         m,
        Number*       g
    );

    /** Method to return:
    *   1) The structure of the jacobian (if "values" is NULL)
    *   2) The values of the jacobian (if "values" is not NULL)
    */
    virtual bool eval_jac_g(
        Index         n,
        const Number* x,
        bool          new_x,
        Index         m,
        Index         nele_jac,
        Index*        iRow,
        Index*        jCol,
        Number*       values
    );

    /** Method to return:
    *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
    *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
    */
    virtual bool eval_h(
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
    );

    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(
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
    );
    //@}
    // JOINT LIMIT
    // void compute_max_min_states(const Number* k);

    TYPE solution[num_joints];

    TYPE* A;
    TYPE* b;

private:
    /**@name Methods to block default compiler methods.
    *
    * The compiler automatically generates the following three methods.
    *  Since the default compiler implementation is generally not what
    *  you want (for all but the most simple classes), we usually
    *  put the declarations of these methods in the private section
    *  and never implement them. This prevents the compiler from
    *  implementing an incorrect "default" behavior without us
    *  knowing. (See Scott Meyers book, "Effective C++")
    */
    //@{
    armtd_NLP(
       const armtd_NLP&
    );

    armtd_NLP& operator=(
       const armtd_NLP&
    );
    uint32_t n_joints;

    // initial condition
    TYPE* qpos;

    TYPE* qvel;

    // desired target
    TYPE* qgoal;
    
    uint32_t n_obs;
    
    // initial guess and g_k
    TYPE* initial_guess;
    TYPE* g_k;

    // keep track of current evaluation point
    Number current_x[n_joints] = { 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0 };

    TYPE* constraints_FO;
    TYPE* jacobian_FO;

    //@}
};

#endif