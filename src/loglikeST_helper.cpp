#include "loglikeST_helper.h"
#include <cstddef>  //std::size_t

void loglikeST_checkSize(const Eigen::Ref<const Eigen::VectorXd>& x_in,  //parameters
                         const Eigen::Ref<const Eigen::VectorXd>& x_fixed,       //fixed parameters
                         char type, //type of likelihood to compute (full, profile, reml)
                         const Eigen::Ref<const Eigen::VectorXd>& Y_in,          //observations
                         const Eigen::Ref<const Eigen::SparseMatrix<double>>& F, //temporal trends
                         const Eigen::Ref<const Eigen::MatrixXd>& M_FX, //LUR and ST covariates
                         const std::vector<std::string>& type_B,  //covariance function for each B-field
                         const Eigen::Ref<const Eigen::VectorXi>& n_cov_pars,       //number of covariance parameters
                         const Eigen::Ref<const Eigen::MatrixXd>& dist_B, //distance matrix for B-fields
                         const Eigen::Ref<const Eigen::MatrixXd>& dist_nu,
                         const Eigen::Ref<const Eigen::VectorXi>& ind){
  //elements that should equal length(Y_in)
  Eigen::Index n = Y_in.size();
  //Does length(Y_in) == F.rows, M_FX.rows, ind.size (i.e. SigmaNu of correct size)
  if( n != F.rows() || n != M_FX.rows() || n != ind.size() ){
    Rcpp::stop("length(Y_in)=%u should match dim(F)[1]=%u, dim(M_FX)[1]=%u, length(ind)=%u", 
               n, F.rows(), M_FX.rows(), ind.size());
  }

  //elements that should equal F.cols()
  //Does F.cols() == dist_B.rows()*type_B.size() (i.e. size og SigmaB)
  if( F.cols() != static_cast<Eigen::Index>(dist_B.rows()*type_B.size()) ){
    Rcpp::stop("dim(F)[2]=%u should match dim(dist_B)[1]*length(type_B)=%u*%u=%u", 
               F.cols(), dist_B.rows(), type_B.size(), dist_B.rows()*type_B.size());
  }
  
  //are distance matrices of equal size and square
  if( dist_B.rows()!=dist_B.cols() || dist_B.rows()!=dist_nu.rows() || dist_B.rows()!=dist_nu.cols() ){
    Rcpp::stop("dist_B and dist_Nu should be square and of equal size.");
  }
  
  //size of parameter vector matches number of B-fields (pars+nugget for each beta field and the nu-field)
  if( n_cov_pars.size() != static_cast<Eigen::Index>(2*type_B.size()+2) ){
    Rcpp::stop("length(n_cov_pars) != %u", 2*type_B.size()+2);
  }
  
  //total number of parameters
  n = n_cov_pars.sum();
  if( type=='f' ){ //add regression parameters for full-likelihood
    n += M_FX.cols();
  }
  if( n!=x_fixed.size() ){
    Rcpp::stop("For type=='%c' expected length(x_fixed)=%u", type, n);
  }
  
  //remaining parameter sizes are checked when extrating parameters in loglikeST_getPars(...)
  return;
}//void loglikeST_checkSize

Eigen::VectorXd loglikeST_fix_x(const Eigen::Ref<const Eigen::VectorXd>& x_in,
                                const Eigen::Ref<const Eigen::VectorXd>& x_fixed){
  //Does is.na(x_fixed) match length(x_in)
  if( x_in.size() != x_fixed.unaryExpr(&R_IsNA).sum() ){
    Rcpp::stop("%u NA value(s) in x_fixed, length(x_in)=%u does not match this.",
         x_fixed.unaryExpr(&R_IsNA).sum(), x_in.size() );
  }
  
  Eigen::VectorXd x = x_fixed;
  for(Eigen::Index i=0, j=0; i<x.size(); ++i){
    if( R_IsNA(x(i)) ){
      x(i) = x_in(j);
      ++j;
    }
  }
  
  return x;
}//Eigen::VectorXd loglikeST_fix_x

void loglikeST_getPars(const Eigen::Ref<const Eigen::VectorXd>& x,
                       char type, //type of likelihood to compute (full, profile, reml)
                       const Eigen::Ref<const Eigen::MatrixXd>& M_FX, //LUR and ST covariates
                       const std::vector<std::string>& type_B,  //covariance function for each B-field
                       const Eigen::Ref<const Eigen::MatrixXd>& nugget_matrix, //design matrix for nugget in nu-field
                       const Eigen::Ref<const Eigen::VectorXi>& n_cov_pars,       //number of covariance parameters
                       Eigen::VectorXd& gamma_alpha,
                       Eigen::VectorXd& par_B, Eigen::VectorXd& nugget_B,
                       Eigen::VectorXd& par_nu, Eigen::VectorXd& nugget_nu,
                       double& random_effect){
  //Extract parameters
  Eigen::Index offset = 0;
  //if type=='f' the first parameters are gamma and alpha.
  if( type=='f' ){ 
    gamma_alpha = x.segment(offset, M_FX.cols());
    offset += M_FX.cols();
  }
  
  //followed by parameters for the sigmaB-matrices.
  Eigen::Index N_pars_B = 0;
  for(std::size_t i=0; i<type_B.size(); ++i){ N_pars_B += n_cov_pars(2*i); }
  
  par_B = Eigen::VectorXd::Zero( N_pars_B );
  nugget_B = Eigen::VectorXd::Zero( type_B.size() );
  Eigen::Index j=0;
  
  for(std::size_t i=0; i<type_B.size(); ++i){
    par_B.segment(j,n_cov_pars(2*i)) = x.segment(offset,n_cov_pars(2*i)).array().exp(); //log-scale for original parameter
    offset += n_cov_pars(2*i);
    j += n_cov_pars(2*i);
    if( n_cov_pars(2*i+1)!=0 ){
      nugget_B(i) = exp( x(offset) ); //log-scale for original parameter
      offset++;
    }
  }
  
  //and finally parameters for sigmaNu.
  par_nu = x.segment(offset, n_cov_pars(2*type_B.size())).array().exp();
  offset += n_cov_pars(2*type_B.size());
  if( nugget_matrix.cols()==0 ){
    nugget_nu = Eigen::VectorXd::Zero( nugget_matrix.rows()  );
  }else{
    nugget_nu.noalias() = nugget_matrix * x.segment(offset, nugget_matrix.cols());
    nugget_nu = nugget_nu.array().exp();
  }
  offset += nugget_matrix.cols();
  //should now have offset==x.size() or offset+1==x.size(); otherwise we have a size missmatch
  if( (offset+1)==x.size() ){
    random_effect = exp(x(offset));
  }else if( offset==x.size() ){
    random_effect = 0;
  }else{
    Rcpp::stop("Unexpected size missmatch in x_fixed");
  }
  
  return;
}//void loglikeST_getPars