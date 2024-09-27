#ifndef LOGLIKEST_R_H
#define LOGLIKEST_R_H

#include <RcppEigen.h>
#include <string>   //std::string
#include <vector>   //std::vector

// [[Rcpp::export]]
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
                     );

// [[Rcpp::export]]
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
);


// [[Rcpp::export]]
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
);


#endif /* LOGLIKEST_R_H */