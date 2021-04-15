#include <RcppEigen.h>

//Compute distance matrix for a a matrix coord
// [[Rcpp::export]]
Eigen::MatrixXd dist_Cpp(const Eigen::Map<Eigen::MatrixXd>& coord){
  //create return matrix
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(coord.rows(),coord.rows());
  //compute lower triangular part (diagonal is zero)
  for(Eigen::Index i=0; i<coord.rows(); ++i){
    for(Eigen::Index j=i+1; j<coord.rows(); ++j){
      res(j,i) = (coord.row(j)-coord.row(i)).norm();
    }
  }
  //complete the matrix as res = res+res'
  res += res.transpose();
  
  //return the result
  return res;
}//dist_Cpp

//Compute cross-distance matrix between coord1 and coord2
// [[Rcpp::export]]
Eigen::MatrixXd crossDist_Cpp(const Eigen::Map<Eigen::MatrixXd>& coord1, 
                              const Eigen::Map<Eigen::MatrixXd>& coord2){
  if(coord1.cols() != coord2.cols()){
    Rcpp::stop("coord1 and coord2 should have the same number of columns.");
  }

  //create return matrix
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(coord1.rows(),coord2.rows());
  //compute complete matrix (no-symmetry due to cross-distance)
  for(Eigen::Index i=0; i<coord2.rows(); ++i){
    for(Eigen::Index j=0; j<coord1.rows(); ++j){
      res(j,i) = (coord1.row(j)-coord2.row(i)).norm();
    }
  }

  //return the result
  return res;
}//crossDist_Cpp