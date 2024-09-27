#ifndef MAKESIGMA_H
#define MAKESIGMA_H

#include <RcppEigen.h>
#include <string>  //std::string
#include <vector>  //std::vector

Eigen::SparseMatrix<double> makeSigmaB(const Eigen::Ref<const Eigen::VectorXd>& par,
                                       const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                       const std::vector<std::string>& type, 
                                       const Eigen::Ref<const Eigen::VectorXd>& nugget,
                                       bool symmetry,
                                       const Eigen::Ref<const Eigen::VectorXi>& ind2_to_1);

Eigen::SparseMatrix<double> makeSigmaNu(const Eigen::Ref<const Eigen::VectorXd>& par,
                                        const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                        const std::string& type, 
                                        const Eigen::Ref<const Eigen::VectorXd>& nugget,
                                        double random_effect, bool symmetry,
                                        const Eigen::Ref<const Eigen::VectorXi>& blocks1,
                                        const Eigen::Ref<const Eigen::VectorXi>& blocks2,
                                        const Eigen::Ref<const Eigen::VectorXi>& ind1,
                                        const Eigen::Ref<const Eigen::VectorXi>& ind2,
                                        const Eigen::Ref<const Eigen::VectorXi>& ind2_to_1);
  
#endif /* MAKESIGMA_H */