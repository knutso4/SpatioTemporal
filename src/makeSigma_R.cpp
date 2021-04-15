#include "makeSigma.h"

// [[Rcpp::export]]
Eigen::SparseMatrix<double> makeSigmaB_cpp(const Eigen::Map<Eigen::VectorXd>& par,
                                           const Eigen::Map<Eigen::MatrixXd>& dist, 
                                           const std::vector<std::string>& type, 
                                           const Eigen::Map<Eigen::VectorXd>& nugget,
                                           bool symmetry,
                                           const Eigen::Map<Eigen::VectorXi>& ind2_to_1){
  return makeSigmaB(par, dist, type, nugget, symmetry, ind2_to_1);
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> makeSigmaNu_cpp(const Eigen::Map<Eigen::VectorXd>& par,
                                            const Eigen::Map<Eigen::MatrixXd>& dist, 
                                            const std::string& type, 
                                            const Eigen::Map<Eigen::VectorXd>& nugget,
                                            double random_effect, bool symmetry,
                                            const Eigen::Map<Eigen::VectorXi>& blocks1,
                                            const Eigen::Map<Eigen::VectorXi>& blocks2,
                                            const Eigen::Map<Eigen::VectorXi>& ind1,
                                            const Eigen::Map<Eigen::VectorXi>& ind2,
                                            const Eigen::Map<Eigen::VectorXi>& ind2_to_1){
  return makeSigmaNu(par, dist, type, nugget, random_effect, symmetry, 
                     blocks1, blocks2, ind1, ind2, ind2_to_1);
  
}
