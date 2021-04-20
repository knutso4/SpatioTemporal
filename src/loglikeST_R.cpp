#include "matrix_helper.h"
#include "loglikeST_helper.h"
#include "loglikeST_R.h"
#include "makeSigma.h"
#include <limits> //std::numeric_limits<double>::lowest()

#include <ctime>


//Note that function calling sequence still needs full names.
//Rcpp functions used
using Rcpp::stop;

//Eigen functions used
using Eigen::Index;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::VectorXi;

//define a few types for easier coding
typedef Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> LLT_Sparse_NoReorder;
typedef Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> LDLT_Sparse_NoReorder;

double getWallTime(){
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return static_cast<double>(clock()) / static_cast<double>(CLOCKS_PER_SEC);
#endif
}


double loglikeST_cpp(const Eigen::Map<Eigen::VectorXd>& x_in,  //parameters
                     const Eigen::Map<Eigen::VectorXd>& x_fixed,       //fixed parameters
                     char type, //type of likelihood to compute (full, profile, reml)
                     const Eigen::Map<Eigen::VectorXd>& Y_in,          //observations
                     const Eigen::Map<Eigen::SparseMatrix<double>>& F, //temporal trends
                     const Eigen::Map<Eigen::MatrixXd>& M_FX, //LUR and ST covariates
                     const std::vector<std::string>& type_B,  //covariance function for each B-field
                     const std::string& type_nu,              //covariance function for nu-field
                     const Eigen::Map<Eigen::MatrixXd>& nugget_matrix, //design matrix for nugget in nu-field
                     const Eigen::Map<Eigen::VectorXi>& n_cov_pars,       //number of covariance parameters
                     const Eigen::Map<Eigen::MatrixXd>& dist_B, //distance matrix for B-fields
                     const Eigen::Map<Eigen::MatrixXd>& dist_nu, //distance matrix for Nu-fields
                     const Eigen::Map<Eigen::VectorXi>& blocks, //block sizes for SigmaNu
                     const Eigen::Map<Eigen::VectorXi>& ind     //index vector for SigmaNu
                       ){

  //sanity check of sizes
  loglikeST_checkSize(x_in, x_fixed, type, Y_in, F, M_FX, type_B, n_cov_pars, dist_B, dist_nu, ind);
  
  //combine x and x_fixed into a single parameter vector
  VectorXd x = loglikeST_fix_x(x_in, x_fixed);
  //...and extract parameters
  VectorXd gamma_alpha, par_B, nugget_B, par_nu, nugget_nu;
  double random_effect;
  loglikeST_getPars(x, type, M_FX, type_B, nugget_matrix, n_cov_pars,
                    gamma_alpha, par_B, nugget_B, par_nu, nugget_nu, random_effect);
  
  //compute Choleskey factorization of SigmaB and SigmaNu.
  //Third template parameter is used to enforce natural ordering. 
  //Enforcing natural order makes sense due to the block structure of the matrices
  LLT_Sparse_NoReorder L_B(makeSigmaB(par_B, dist_B, type_B, nugget_B, true, VectorXi::Zero(0)));
  LLT_Sparse_NoReorder L_nu(makeSigmaNu(par_nu, dist_nu, type_nu, nugget_nu, random_effect,
                            true, blocks, blocks, ind, ind, VectorXi::Zero(0)));
  
  //check if matrices are positive definite
  if( !check_LLT(L_B) || !check_LLT(L_nu) ){
    return std::numeric_limits<double>::lowest();
  }
  //determinants for SigmaNu and SigmaB
  double l = -(logDet_LLT(L_B) + logDet_LLT(L_nu));
  
  //compute SigmaB_Y
  LLT<MatrixXd> L_SigmaB_Y = compute_L_Sigma_B_Y< LLT<MatrixXd> >(F, L_B, L_nu);
  
  //check if SigmaB_Y is positive definite
  if( !check_LLT(L_SigmaB_Y) ){
    return std::numeric_limits<double>::lowest();
  }
  //and compute determinante
  l -= logDet_LLT(L_SigmaB_Y);

  //compute alpha and gamma if needed
  if( type!='f' ){
    Eigen::LLT<Eigen::MatrixXd> L_Sigma_GA_Y;
    if( !loglikeST_GammaAlpha(Y_in, F, M_FX, L_SigmaB_Y, L_nu, gamma_alpha, L_Sigma_GA_Y) ){
      //choleskey of L_Sigma_GA_Y failed (not pos def)
      return std::numeric_limits<double>::lowest();
    }
    if( type=='r' ){
      //REML likelihood; add contribution from determinant of Sigma_GA_Y.
      l -= logDet_LLT( L_Sigma_GA_Y );
    }
  }//if( type!='f' )

  //and compute Y - M*gamma - F*X*alpha as needed by the quadratic form
  VectorXd Y = Y_in  - M_FX * gamma_alpha;
  //and compute Sigma_nu^-1 * Y
  VectorXd iS_Y = L_nu.solve(Y);

  l -= iS_Y.cwiseProduct( Y - F * L_SigmaB_Y.solve( F.transpose() * iS_Y ) ).array().sum();
  /*
  //compute quadratic form (Y' * Sigma_nu^-1 * Y)
  l -= L_nu.solve(Y).cwiseProduct(Y).array().sum();
  //(Y' * Sigma_nu^-1 * F * Sigma_B_Y^-1 * F' * Sigma_nu^-1 * Y)
  VectorXd Y_iS_F = Y.transpose() * iS_F;
  l += L_SigmaB_Y.solve( Y_iS_F ).cwiseProduct( Y_iS_F ).array().sum();
  */
  
  //return completed log-likelihood
  return l/2.0;
}//loglikeST_cpp

double loglikeSTnaive_cpp(const Eigen::Map<Eigen::VectorXd>& x_in,  //parameters
                          const Eigen::Map<Eigen::VectorXd>& x_fixed,       //fixed parameters
                          char type, //type of likelihood to compute (full, profile, reml)
                          const Eigen::Map<Eigen::VectorXd>& Y_in,          //observations
                          const Eigen::Map<Eigen::SparseMatrix<double>>& F, //temporal trends
                          const Eigen::Map<Eigen::MatrixXd>& M_FX, //LUR and ST covariates
                          const std::vector<std::string>& type_B,  //covariance function for each B-field
                          const std::string& type_nu,              //covariance function for nu-field
                          const Eigen::Map<Eigen::MatrixXd>& nugget_matrix, //design matrix for nugget in nu-field
                          const Eigen::Map<Eigen::VectorXi>& n_cov_pars,       //number of covariance parameters
                          const Eigen::Map<Eigen::MatrixXd>& dist_B, //distance matrix for B-fields
                          const Eigen::Map<Eigen::MatrixXd>& dist_nu, //distance matrix for Nu-fields
                          const Eigen::Map<Eigen::VectorXi>& blocks, //block sizes for SigmaNu
                          const Eigen::Map<Eigen::VectorXi>& ind     //index vector for SigmaNu
                            ){
  //sanity check of sizes
  loglikeST_checkSize(x_in, x_fixed, type, Y_in, F, M_FX, type_B, n_cov_pars, dist_B, dist_nu, ind);
  
  //combine x and x_fixed into a single parameter vector
  VectorXd x=loglikeST_fix_x(x_in, x_fixed);
  //...and extract parameters
  VectorXd gamma_alpha, par_B, nugget_B, par_nu, nugget_nu;
  double random_effect;
  loglikeST_getPars(x, type, M_FX, type_B, nugget_matrix, n_cov_pars,
                    gamma_alpha, par_B, nugget_B, par_nu, nugget_nu, random_effect);
  
  //extract observations (copy needed due to type=='f'; hopefully optimized out for other cases)
  VectorXd Y = Y_in;
  //compute Y - (M*gamma + F*X*alpha)
  if( type=='f' ){
    Y -= M_FX * gamma_alpha;
  }//if( type=='f' ){
  
  //compute covariance matrices
  MatrixXd sigmaFull = F * makeSigmaB(par_B, dist_B, type_B, nugget_B, true, 
                                      VectorXi::Zero(0)).selfadjointView<Lower>() * F.transpose();
  sigmaFull += makeSigmaNu(par_nu, dist_nu, type_nu, nugget_nu, random_effect, true, 
                           blocks, blocks, ind, ind, VectorXi::Zero(0));
  
  //compute choleskey factor of sigmaFull
  LLT<MatrixXd> RsigmaFull(sigmaFull);
  
  //check if matrix is positive definite
  if( !check_LLT(RsigmaFull) ){
    return std::numeric_limits<double>::lowest();
  }
  
  //start computing likelihood
  // -log(det(sigma.nu)^.5)
  double l = -logDet_LLT(RsigmaFull);
  //calculate if(type=="f") inv(R)'*(Y-mean.val) else inv(R)'*Y
  //if(type=="f")  -1/2 (Y-mean)' * inv(sigma.nu) * (Y-mean)
  //     else      -1/2 Y' * inv(sigma.nu) * Y
  MatrixXd iL_Y = RsigmaFull.matrixL().solve(Y);
  l -= iL_Y.squaredNorm();
  
  //components for profile or REML computations
  if( type!='f' ){
    //calculate inv(L)*M_FX
    MatrixXd iL_F = RsigmaFull.matrixL().solve(M_FX);
    //calculate Ftmp'*inv(Sigma)*Y
    MatrixXd FY = iL_F.transpose() * iL_Y;
    //calculate chol([FX M]'*invSigma*[FX M])
    LLT<MatrixXd> RsigmaAlt( iL_F.transpose() * iL_F );
    //check if matrix is positive definite
    if( !check_LLT(RsigmaAlt) ){
      return std::numeric_limits<double>::lowest();
    }
    //+1/2 FY' * inv(sigma.alt) * FY
    l += RsigmaAlt.matrixL().solve(FY).squaredNorm();
    //-log(det(sigma.alt)^.5)
    if( type=='r' ){
      l -= logDet_LLT(RsigmaAlt);
    }//type=='r'
  }//if( type!='f' )
  
  return l/2.0;
}//loglikeSTnaive_cpp












Eigen::MatrixXd loglikeST_test_cpp(const Eigen::Map<Eigen::VectorXd>& x_in,  //parameters
                     const Eigen::Map<Eigen::VectorXd>& x_fixed,       //fixed parameters
                     char type, //type of likelihood to compute (full, profile, reml)
                     const Eigen::Map<Eigen::VectorXd>& Y_in,          //observations
                     const Eigen::Map<Eigen::SparseMatrix<double>>& F, //temporal trends
                     const Eigen::Map<Eigen::MatrixXd>& M_FX, //LUR and ST covariates
                     const std::vector<std::string>& type_B,  //covariance function for each B-field
                     const std::string& type_nu,              //covariance function for nu-field
                     const Eigen::Map<Eigen::MatrixXd>& nugget_matrix, //design matrix for nugget in nu-field
                     const Eigen::Map<Eigen::VectorXi>& n_cov_pars,       //number of covariance parameters
                     const Eigen::Map<Eigen::MatrixXd>& dist_B, //distance matrix for B-fields
                     const Eigen::Map<Eigen::MatrixXd>& dist_nu, //distance matrix for Nu-fields
                     const Eigen::Map<Eigen::VectorXi>& blocks, //block sizes for SigmaNu
                     const Eigen::Map<Eigen::VectorXi>& ind     //index vector for SigmaNu
){
  
  //sanity check of sizes
  loglikeST_checkSize(x_in, x_fixed, type, Y_in, F, M_FX, type_B, n_cov_pars, dist_B, dist_nu, ind);
  
  //combine x and x_fixed into a single parameter vector
  VectorXd x = loglikeST_fix_x(x_in, x_fixed);
  //...and extract parameters
  VectorXd gamma_alpha, par_B, nugget_B, par_nu, nugget_nu;
  double random_effect;
  loglikeST_getPars(x, type, M_FX, type_B, nugget_matrix, n_cov_pars,
                    gamma_alpha, par_B, nugget_B, par_nu, nugget_nu, random_effect);
  
  double start = getWallTime();
  
  //compute Choleskey factorisation of SigmaB and SigmaNu.
  //Third template parameter is used to enforce natural ordering. 
  //This makes sense due to the block structure of the matricies
  LLT_Sparse_NoReorder L_B(makeSigmaB(par_B, dist_B, type_B, nugget_B, true, VectorXi::Zero(0)));
  LLT_Sparse_NoReorder L_nu(makeSigmaNu(par_nu, dist_nu, type_nu, nugget_nu, random_effect,
                                        true, blocks, blocks, ind, ind, VectorXi::Zero(0)));

  Rprintf("\nTime chols: %.5f s\n", getWallTime()-start);
  start = getWallTime();
  
  LLT<MatrixXd> F_iS_F = compute_L_Sigma_B_Y< LLT<MatrixXd> >(F, L_B, L_nu);
    
/*  MatrixXd F_iS_F = F.transpose() * L_nu.solve(F);
  Eigen::SparseMatrix<double> spIdent(L_B.cols(),L_B.cols());
  spIdent.setIdentity();
  F_iS_F += L_B.solve(spIdent);
*/
  
  Rprintf("\nTime prods: %.5f s\n", getWallTime()-start);
  
  return F_iS_F.matrixL();
  /*
  //  F_iS_F.triangularView<Eigen::Lower>() = F.transpose() * iS_F;
  //compute SigmaB_Y
  LLT<MatrixXd> L_SigmaB_Y( F_iS_F + L_B.solve(MatrixXd::Identity(L_B.cols(),L_B.cols())) );
  
  //check if SigmaB_Y is positive definite
  if( !check_LLT(L_SigmaB_Y) ){
    return std::numeric_limits<double>::lowest();
  }
  //and compute determinante
  l -= logDet_LLT(L_SigmaB_Y);
  
  //compute alpha and gamma if needed
  if( type!='f' ){
    Eigen::LLT<Eigen::MatrixXd> L_Sigma_GA_Y;
    if( !loglikeST_GammaAlpha(Y_in, F, X, M, iS_F, F_iS_F, L_SigmaB_Y, L_nu, 
                              gamma, alpha, L_Sigma_GA_Y) ){
      //choleskey of L_Sigma_GA_Y failed (not pos def)
      return std::numeric_limits<double>::lowest();
    }
    if( type=='r' ){
      //REML likelihood; add contribution from determinant of Sigma_GA_Y.
      l -= logDet_LLT( L_Sigma_GA_Y );
    }
  }//if( type!='f' )
  
  //and compute Y - M*gamma - F*X*alpha as needed by the quadratic form
  VectorXd Y = Y_in  - F * X * alpha;
  if( M.cols()!=0 ){ Y -= M * gamma; }
  
  //compute quadratic form (Y' * Sigma_nu^-1 * Y)
  l -= L_nu.solve(Y).cwiseProduct(Y).array().sum();
  //(Y' * Sigma_nu^-1 * F * Sigma_B_Y^-1 * F' * Sigma_nu^-1 * Y)
  VectorXd Y_iS_F = Y.transpose() * iS_F;
  l += L_SigmaB_Y.solve( Y_iS_F ).cwiseProduct( Y_iS_F ).array().sum();
  
  //return completed log-likelihood
  return l/2.0;
  */
}//loglikeST_internal
