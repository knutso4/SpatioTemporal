#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#include <RcppEigen.h>

//For use of Eigen:Ref see
//https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class
//Short version use (e.g. T=Eigen::VectorXd or T=MatrixXd):
// const Ref<const T>& for a const reference.
// T& for a writable reference (note that recommended Eigen is to use Ref<T>, but this gives me memory leaks...)
//For the Choleskey factors we use a template format const T& to allow for either sparse or dense Choleskey factors

//Check Choleskey factor for a LDLT_transformation
template<typename T>
bool check_LDLT(const T& LDLT){
  return LDLT.info()==Eigen::Success && (LDLT.vectorD().array()>0).all();
}//check_LDLT

//compute log determinante
template<typename T>
double logDet_LDLT(const T& LDLT){
  return LDLT.vectorD().array().log().sum();
}//logDet_LDLT

template<typename T>
bool check_LLT(const T& LLT){
  return LLT.info()==Eigen::Success;
}//check_LDLT

//compute log determinante
template<typename T>
double logDet_LLT(const T& LLT){
  return 2.0 * LLT.matrixL().selfadjointView().diagonal().array().log().sum();
}//logDet_LDLT

template<typename LLT_RET_TYPE, typename LLT_IN1_TYPE, typename LLT_IN2_TYPE>
LLT_RET_TYPE compute_L_Sigma_B_Y(const Eigen::Ref<const Eigen::SparseMatrix<double>>& F, //temporal trends
                    const LLT_IN1_TYPE& L_B, //sparse choleskey( sigma_B )
                    const LLT_IN2_TYPE& L_nu){ //sparse choleskey( sigma_nu )
  //compute F' * sigma_nu^-1 * F
  Eigen::MatrixXd F_iS_F = F.transpose() * L_nu.solve(F);
  //create a sparse identity matrix
  Eigen::SparseMatrix<double> spIdent(L_B.cols(),L_B.cols());
  spIdent.setIdentity();
  //compute F' * sigma_nu^-1 * F + sigma_B^-1
  F_iS_F += L_B.solve(spIdent);
  //and compute (dense) choleskey factor based on this
  LLT_RET_TYPE L_SigmaB_Y( F_iS_F );
  //return the choleskey factor
  return L_SigmaB_Y;
}//LLT_TYPE compute_L_Sigma_B_Y

template<typename LLT_IN1_TYPE, typename LLT_IN2_TYPE>
bool loglikeST_GammaAlpha(const Eigen::Ref<const Eigen::VectorXd>& Y,          //observations
                          const Eigen::Ref<const Eigen::SparseMatrix<double>>& F, //temporal trends
                          const Eigen::Ref<const Eigen::MatrixXd>& M_FX, //LUR and ST covariates
                          const LLT_IN1_TYPE& L_SigmaB_Y, //choleskey( sigma_B_Y )
                          const LLT_IN2_TYPE& L_nu, //sparse choleskey( sigma_nu )
                          Eigen::VectorXd& gamma_alpha, //return values for E([gamma,alpha]| Y)
                          Eigen::LLT<Eigen::MatrixXd>& L_Sigma_GA_Y //choleskey factor of V([gamma,alpha]| Y)^-1
){
  //precompute sigma_nu^-1 * [M F*X]
  Eigen::MatrixXd iS_MFX = L_nu.solve(M_FX);
  
  //start computation of Sigma_GA_Y
  // Sigma_GA_Y = M_FX' * sigma_nu^-1 * M_FX - M_FX' * sigma_nu^-1 * F * SigmaB_Y^-1 * F' * sigma_nu^-1 * M_FX
  //            = iS_MFX' * ( M_FX -  F * SigmaB_Y^-1 * F' * iS_MFX)
  Eigen::MatrixXd MFX_FiSBYFtMFX = M_FX - F * L_SigmaB_Y.solve( F.transpose() * iS_MFX );
//  Eigen::MatrixXd MFX_FiSBYFtMFX = compute_Sn_iSX(M_FX, F, L_SigmaB_Y, L_nu, iS_MFX);
    
  //compute choleskey factor of Sigma_GA_Y matrix
  L_Sigma_GA_Y.compute( iS_MFX.transpose() * MFX_FiSBYFtMFX );
  
  //check choleskey factorisation
  if( !check_LLT(L_Sigma_GA_Y) ){ return false; }
  
  //compute M_FX' * sigma_tilde^-1 * Y = M_FX' * sigma_nu^-1 * Y - M_FX' * sigma_nu^-1 * F * SigmaB_Y^-1 * F' * sigma_nu^-1 * Y
  //(M_FX -  F * SigmaB_Y^-1 * F' * iS_MFX)' * iS_Y' 
  gamma_alpha = MFX_FiSBYFtMFX.transpose() * L_nu.solve(Y);
  
  //solve Sigma_GA_Y^-1 [gamma, alpha]. Doing so inplace.
  L_Sigma_GA_Y.solveInPlace(gamma_alpha);

  //and return
  return true;
}//bool loglikeST_GammaAlpha

#endif /* MATRIX_HELPER_H */