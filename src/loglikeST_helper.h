#ifndef LOGLIKEST_HELPER_H
#define LOGLIKEST_HELPER_H

#include <RcppEigen.h>
#include <string>   //std::string
#include <vector>   //std::vector

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
                         const Eigen::Ref<const Eigen::VectorXi>& ind);

Eigen::VectorXd loglikeST_fix_x(const Eigen::Ref<const Eigen::VectorXd>& x_in,
                                const Eigen::Ref<const Eigen::VectorXd>& x_fixed);

void loglikeST_getPars(const Eigen::Ref<const Eigen::VectorXd>& x,
                       char type, //type of likelihood to compute (full, profile, reml)
                       const Eigen::Ref<const Eigen::MatrixXd>& M_FX, //LUR and ST covariates
                       const std::vector<std::string>& type_B,  //covariance function for each B-field
                       const Eigen::Ref<const Eigen::MatrixXd>& nugget_matrix, //design matrix for nugget in nu-field
                       const Eigen::Ref<const Eigen::VectorXi>& n_cov_pars,       //number of covariance parameters
                       Eigen::VectorXd& gamma_alpha,
                       Eigen::VectorXd& par_B, Eigen::VectorXd& nugget_B,
                       Eigen::VectorXd& par_nu, Eigen::VectorXd& nugget_nu,
                       double& random_effect);

#endif /* LOGLIKEST_HELPER_H */