###################################################
## INTERFACES FOR Rcpp - COVARIANCE CONSTRUCITON ##
###################################################
##Functions in this file:
## makeSigmaB     EX:ok
### diffSigmaB     TODO
## makeSigmaNu    EX:ok
### diffSigmaNu    TODO
## parsCovFuns    EX:ok
## namesCovFuns   EX:ok
## evalCovFuns    EX:ok

##Provides interface functions against Rcpp code
##see also src/covf_R.cpp
##             covf.h
##             covf.cpp
##             makeSigma_R.cpp

##' Function that creates a block covariance matrix with equal sized blocks.
##' Used to construct the Sigma_B matrix.
##'
##' Any parameters given as scalars will be \code{rep}-ed to match
##' \code{length(pars)}.
##' 
##' @title Create Block Covariance Matrix (Equal Block Sizes)
##' @param pars List of parameters for each block; if not a list a single
##'   block matrix is assumed. Should match parameters suggested by
##'   \code{\link{parsCovFuns}}.
##' @param dist Distance matrix.
##' @param type Name(s) of covariance functions, see
##'   \code{\link{namesCovFuns}}.
##' @param nugget Vector of nugget(s) to add to the diagonal of each matrix.
##' @param symmetry \code{TRUE}/\code{FALSE} flag if the \code{dist} is
##'   symmetric, resulting in a symmetric covariance matrix.
##' @param ind2.to.1 Vectors, that for each index along the second dimension
##'   gives a first dimension index, used only if \code{symmetry=FALSE}
##'   to determine which covariances should have an added nugget (co-located
##'   sites).
##' @param diff Vector with two components indicating with respect to which
##'   parameter(s) that first and/or second derivatives should be
##'   computed. E.g. \code{diff=c(0,0)} indicates no derivatives,
##'   \code{diff=c(1,0)} indicates first derivative wrt the first parameter,
##'   \code{diff=c(1,2)} indicates second cross derivative wrt the first and
##'   second parameters, etc.
##' @return Block diagonal sparse matrix.
##' 
##' @example Rd_examples/Ex_makeSigmaB.R
##' 
##' @author Johan Lindstrom
##' @family block matrix functions
##' @family covariance functions
##' @export
makeSigmaB <- function(pars, dist, type="exp", nugget=0,
                       symmetry=dim(dist)[1]==dim(dist)[2],
                       ind2.to.1=1:dim(dist)[2], diff=0){
  if( !is.list(pars) ){
    ##pars not a list, assuming that we only have one block
    n.blocks <- 1
  }else{  
    n.blocks <- length(pars)
  }
  ##repeat length 1 blocks to suitable size.
  if( length(type)==1 ){
    type <- rep(type, n.blocks)
  }
  if( length(nugget)==1 ){
    nugget <- rep(nugget, n.blocks)
  }
  
  ##call C-code, internal error-checking
  tmp <- makeSigmaB_cpp(unlist(pars), dist, type, nugget, as.logical(symmetry),
                             as.integer(ind2.to.1))
  return( tmp )
}##function makeSigmaB

##' Function that creates a block covariance matrix with unequally
##' sized blocks. Used to construct the Sigma_nu matrix.
##'
##' @title Create Block Covariance Matrix (Unequal Block Sizes)
##' @param pars Vector of parameters, as suggested by
##'   \code{parsCovFuns}.
##' @param dist Distance matrix.
##' @param type Name of the covariance function to use, see
##'   \code{\link{namesCovFuns}}.
##' @param nugget A value of the nugget or a vector of length
##'   \code{dim(dist)[1]} giving (possibly) location specific nuggets.
##' @param random.effect A constant variance to add to the covariance matrix,
##'   can be interpreted as either and partial sill with infinite
##'   range or as a random effect with variance given by \code{random.effect}
##'   for the mean value.
##' @param symmetry \code{TRUE}/\code{FALSE} flag if the \code{dist} matrix is
##'   symmetric. If also \code{ind1==ind2} and \code{blocks1==blocks2} the
##'   resulting covariance matrix will be symmetric.
##' @param blocks1,blocks2 Vectors with the size(s) of each of the
##'   diagonal blocks, usually \code{\link{mesa.model}$nt}. If \code{symmetry=TRUE}
##'   then \code{blocks2} defaults to \code{blocks1} if missing.
##' @param ind1,ind2 Vectors indicating the location of each element in the
##'   covariance matrix, used to index the \code{dist}-matrix to
##'   determine the distance between locations, usually
##'   \code{\link{mesa.model}$obs$idx}. If \code{symmetry=TRUE}
##'   and then \code{ind2} defaults to \code{ind1} if missing.
##' @param ind2.to.1 Vectors, that for each index along the second dimension,
##'   \code{ind2}, gives a first dimension index, \code{ind1}, used only if
##'   \code{symmetry=FALSE} to determine which covariances should have an
##'   added nugget (collocated sites).
##' @param diff Vector with two components indicating with respect to which
##'   parameter(s) that first and/or second derivatives should be
##'   computed. E.g. \code{diff=c(0,0)} indicates no derivatives,
##'   \code{diff=c(1,0)} indicates first derivative wrt the first parameter,
##'   \code{diff=c(1,2)} indicates second cross derivative wrt the first and
##'   second parameters, etc.
##' @return Block diagonal sparse covariance matrix of size 
##'   \code{length(ind1)}-by-\code{length(ind2)}.
##' 
##' @example Rd_examples/Ex_makeSigmaNu.R
##' 
##' @author Johan Lindstrom
##' @family block matrix functions
##' @family covariance functions
##' @export
makeSigmaNu <- function(pars, dist, type="exp", nugget=0, random.effect=0,
                        symmetry=dim(dist)[1]==dim(dist)[2],
                        blocks1=dim(dist)[1], blocks2=dim(dist)[2],
                        ind1=1:dim(dist)[1], ind2=1:dim(dist)[2], 
                        ind2.to.1=1:dim(dist)[2], diff=0){
  if( missing(blocks2) && symmetry ){
    blocks2 <- blocks1
  }
  if( missing(ind2) && symmetry ){
    ind2 <- ind1
  }
  ##call C-code, internal error-checking
  tmp <- makeSigmaNu_cpp(pars, dist, type, nugget, random.effect, as.logical(symmetry),
                              as.integer(blocks1), as.integer(blocks2), 
                              as.integer(ind1), as.integer(ind2), as.integer(ind2.to.1))
  return( tmp )
}##function makeSigmaNu

##' Provides a list of parameter names for the given covariance function(s),
##' excluding the nugget which is added elsewhere.
##'
##' @title Parameter Names for Covariance Function(s)
##' @param type Name(s) of covariance functions, see \code{\link{namesCovFuns}}.
##' @param list Always return a list (if FALSE returns a vector if possible)
##' @return Character vector with parameter names (excluding the nugget),
##'   \code{NULL} if the name is unknown. Returns a list if type contains
##'   more than one element.
##' 
##' @examples
##'   ##all possible parameters
##'   parsCovFuns()
##'   ##just one covariance function
##'   parsCovFuns("exp")
##' 
##' @author Johan Lindstrom
##' @family covariance functions
##' @export
parsCovFuns <- function(type = namesCovFuns(), list=FALSE){
  if( length(type)==0 ){ stop("'type' has to be of length>0") }
  ##special case for length one type
  if( length(type)==1 && !list){
    return( parsCovFuns_cpp(type[1]) )
  }
  ##o.w. return a list of possible names
  par.names <- vector("list", length(type))
  for(i in 1:length(type)){
    par.names[[i]] <- parsCovFuns_cpp(type[i])
  }
  names(par.names) <- type
  return( par.names )
}##function parsCovFuns

##' Computes covariance functions (excluding nugget) for a given vector or
##' matrix of distances.
##'
##' @title Compute Covariance Function
##' @param type Name of covariance functions, see \code{\link{namesCovFuns}}.
##' @param pars Parameter for the covariance function, see
##'   \code{\link{parsCovFuns}}.
##' @param d Vector/matrix for which to compute the covariance function.
##' @param diff Vector with two components indicating with respect to which
##'   parameter(s) that first and/or second derivatives should be
##'   computed. E.g. \code{diff=c(0,0)} indicates no derivatives,
##'   \code{diff=c(1,0)} indicates first derivative wrt the first parameter,
##'   \code{diff=c(1,2)} indicates second cross derivative wrt the first and
##'   second parameters, etc.
##' @return Covariance function computed for all elements in d.
##' 
## @example Rd_examples/Ex_evalCovFuns.R
##' 
##' @author Johan Lindstrom
##' @family covariance functions
##' @export
evalCovFuns <- function(type="exp", pars=c(1,1), d=seq(0,10,length.out=100), diff=c(0,0)){
  ##call C-code, internal error-checking
  return( evalCovFuns_cpp(type, pars, d, as.integer(diff)) )
}##function evalCovFuns