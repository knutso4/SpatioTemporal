#include <RcppEigen.h>
#include <vector>  //std::vector
#include <algorithm> //std::max

// [[Rcpp::export]]
Eigen::SparseMatrix<double> expandF_cpp(const Eigen::Map<Eigen::MatrixXd>& F, 
                        const Eigen::Map<Eigen::VectorXi>& loc_ind, int n_loc){
  //check dimensions
  if( F.rows() != loc_ind.size() ){
    Rcpp::stop("In 'expandF': dim(F)[1] != length(loc_ind)");
  }
  if( loc_ind.maxCoeff() > n_loc ){
    Rcpp::stop("In 'expandF': max(loc.ind) > n_loc");
  }

  std::vector< Eigen::Triplet<double> > tripletList;
  tripletList.reserve(F.rows()*F.cols());
  for(auto i=0; i<F.cols(); ++i){
    for(auto j=0; j<F.rows(); ++j){
      //add each element of F into the triplet list. Recall that loc_ind is from R using 1-based indecies!!
      tripletList.push_back( Eigen::Triplet<double>(j, loc_ind(j)-1 + i*n_loc, F(j,i)) );
    }
  }
  //create a sparsematrix of suitable size
  Eigen::SparseMatrix<double> m(F.rows(), F.cols()*n_loc);
  //and initialize using the triplet list.
  m.setFromTriplets(tripletList.begin(), tripletList.end());
  
  //return the expanded matrix
  return(m);
}//function expandF_cpp