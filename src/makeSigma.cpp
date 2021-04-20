#include "makeSigma.h"
#include "covf.h"

Eigen::SparseMatrix<double> makeSigmaB(const Eigen::Ref<const Eigen::VectorXd>& par,
                                       const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                       const std::vector<std::string>& type, 
                                       const Eigen::Ref<const Eigen::VectorXd>& nugget,
                                       bool symmetry,
                                       const Eigen::Ref<const Eigen::VectorXi>& ind2_to_1){
  //extract covariance functions
  std::vector<std::unique_ptr<covf_functor>> covf(type.size());
  std::vector<std::size_t> n_pars(type.size());
  for(std::size_t i=0; i<type.size(); ++i){
    covf[i] = select_covf( type[i] );
    if( covf[i]==nullptr ){ 
      Rcpp::stop("Unknown covariance function: %s", type[i]);
    }
    n_pars[i] = covf[i]->nPars();
  }
  //symmetry should be false if dist not square
  symmetry &= dist.rows()==dist.cols();
  
  //Sanity check of input sizes
  if( type.size() != static_cast<std::size_t>(nugget.size()) ){
    Rcpp::stop("Size missmatch: length(type)!=length(nugget)");
  }
  if( par.size() != std::accumulate(n_pars.begin(), n_pars.end(), 0) ){
    Rcpp::stop("Size missmatch: total number of parameters !=%u", 
               std::accumulate(n_pars.begin(), n_pars.end(), 0));
  }
  if( !symmetry && ind2_to_1.size()!=dist.cols() ){
    //only care about ind2_to_1 in non symmetric case
    Rcpp::stop("length(ind2_to_1)!=dim(dist)[2]");
  }
  
  //create empty sparse matrix for return
  Eigen::SparseMatrix<double> sigmaB(dist.rows()*covf.size(), dist.cols()*covf.size());
  //reserve size in the matrix
  sigmaB.reserve( Eigen::VectorXi::Constant(sigmaB.cols(), dist.rows()) );
  //compute covariance matrices for each component and move these into the final sparse matrix
  Eigen::Index offset = 0;
  for(std::size_t i=0; i<covf.size(); ++i){
    //compute covariance matrix for this block
    Eigen::MatrixXd tmp = covf[i]->operator()(par.segment(offset,n_pars[i]), dist);
    //add nugget... 
    if(symmetry){ //to diagonal
      tmp.diagonal().array() += nugget[i];
    }else{ //or as indicated
      //loop over second index (columns)
      for(Eigen::Index k=0; k<tmp.cols(); ++k){
        //and find corresponding first index (note -1 due to R being 1-based index)
        Eigen::Index l = ind2_to_1[k]-1;
        //check if index is valid, and if so add nugget
        if( l>=0 && l<tmp.rows() ){
          tmp(l,k) += nugget[i];
        }
      }
    }
  
    //copy elements into the correct sparse block
    for(Eigen::Index k=0; k<dist.cols(); ++k){
      for(Eigen::Index l=0; l<dist.rows(); ++l){
        sigmaB.insert(l + i*dist.rows(), k + i*dist.cols()) = tmp(l,k);
      }
    }
  
    //increase offset for start of next block
    offset += n_pars[i];
  }//for(std::size_t i=0; i<covf.size(); ++i)
  
  //compute covariance matrix for first component
  return sigmaB;
}//makeSigmaB

Eigen::SparseMatrix<double> makeSigmaNu(const Eigen::Ref<const Eigen::VectorXd>& par,
                                        const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                        const std::string& type, 
                                        const Eigen::Ref<const Eigen::VectorXd>& nugget,
                                        double random_effect, bool symmetry,
                                        const Eigen::Ref<const Eigen::VectorXi>& blocks1,
                                        const Eigen::Ref<const Eigen::VectorXi>& blocks2,
                                        const Eigen::Ref<const Eigen::VectorXi>& ind1,
                                        const Eigen::Ref<const Eigen::VectorXi>& ind2,
                                        const Eigen::Ref<const Eigen::VectorXi>& ind2_to_1){
  //symmetry should be false if dist not square
  symmetry &= dist.rows()==dist.cols();
  
  //extract covariance functions
  std::unique_ptr<covf_functor> covf = select_covf(type);
  if( covf==nullptr ){ 
    Rcpp::stop("Unknown covariance function: %s", type);
  }
  if( par.size() != covf->nPars() ){
    Rcpp::stop("Size missmatch: length(pars)!=%u", covf->nPars());
  }

  //Sanity check of input sizes
  if( nugget.size()!=1 && nugget.size()!=dist.rows() ){
    Rcpp::stop("Nugget needs to contain 1 or dim(dist)[1] elements.");
  }
  //blocks of equal size and size of blocks match index vectors
  if( blocks1.size() != blocks2.size() ){
    Rcpp::stop("length(blocks1)!=length(blocks2)");
  }
  if( blocks1.sum() != ind1.size() ){
    Rcpp::stop("sum(blocks1)!=length(ind1)");
  }
  if( blocks2.sum() != ind2.size() ){
    Rcpp::stop("sum(blocks2)!=length(ind2)");
  }
  //indecies are valid (recall 1-index from R-code!)
  if( ind1.minCoeff()<1 || ind1.maxCoeff()>dist.rows() ){
    Rcpp::stop("min(ind1)<1 or max(ind1)>dim(dist)[1]");
  }
  if( ind2.minCoeff()<1 || ind2.maxCoeff()>dist.cols() ){
    Rcpp::stop("min(ind2)<1 or max(ind2)>dim(dist)[2]");
  }
  if( !symmetry && ind2_to_1.size()!=dist.cols() ){
    //only care about ind2_to_1 in non symmetric case
    Rcpp::stop("length(ind2_to_1)!=dim(dist)[2]");
  }
  
  //Compute the (cross-)covariance matrix for all locations.
  Eigen::MatrixXd tmp = covf->operator()(par, dist);
  //add nugget... 
  if(symmetry){ //to diagonal
    if( nugget.size()==1 ){
      tmp.diagonal().array() += nugget(0);
    }else{
      tmp.diagonal() += nugget;
    }
  }else{ //or as indicated
    //loop over second index (columns)
    for(Eigen::Index k=0; k<tmp.cols(); ++k){
      //and find corresponding first index (note -1 due to R being 1-based index)
      Eigen::Index l = ind2_to_1[k]-1;
      //check if index is valid, and if so add nugget
      if( l>=0 && l<tmp.rows() ){
        if( nugget.size()==1 ){
          tmp(l,k) += nugget(0);
        }else{
          tmp(l,k) += nugget(l);
        }
      }
    }
  }//if(symmetry){...}else{...}

  //create empty sparse matrix for return
  Eigen::SparseMatrix<double> sigmaNu(ind1.size(), ind2.size());
  //reserve size in the matrix
  Eigen::VectorXi nnz_cols(sigmaNu.cols());
  for(Eigen::Index i=0, k=0; i<blocks2.size(); ++i){
    for(Eigen::Index j=0; j<blocks2(i); ++j, ++k){
      nnz_cols(k) = blocks1(i);
    }
  }
  sigmaNu.reserve( nnz_cols );
  //compute covariance matrices for each component and move these into the final sparse matrix
  Eigen::Index offset_rows = 0;
  Eigen::Index offset_cols = 0;
  for(Eigen::Index i=0; i<blocks1.size(); ++i){
    //copy elements into the correct sparse block
    for(Eigen::Index k=0; k<blocks2(i); ++k){
      //recall 1-index in R-code
      Eigen::Index i_col = ind2(k + offset_cols)-1;
      for(Eigen::Index l=0; l<blocks1(i); ++l){
        //recall 1-index in R-code
        sigmaNu.insert(l + offset_rows, k + offset_cols) = tmp(ind1(l + offset_rows)-1, i_col);
//        Rprintf("i: %i; k: %i; l: %i; (%i,%i) = (%i,%i)\n", 
//                i, k ,l, l + offset_rows, k + offset_cols, ind1(l + offset_rows)-1, i_col);
      }
    }
    //incremeant offset by size of latest block
    offset_rows += blocks1(i);
    offset_cols += blocks2(i);
  }//for(Eigen::Index i=0; i<blocks1.size(); ++i)
  
  return sigmaNu;
}//makeSigmaNu_cpp