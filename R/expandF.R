######################################################
## INTERFACES FOR C CODE - SPARSE F TIMES SOMETHING ##
######################################################
##Functions in this file:
## expandF     EX:OK

##' Expands the temporal trends in F to a full matrix (with lots of zeros).
##' Mainly used for testing, and illustration in examples.
##'
##' @title Expand F
##' @param F A (number of obs.) - by - (number of temporal trends) matrix
##'   containing the temporal trends. Usually \code{\link{mesa.model}$F}, where
##'   \code{\link{mesa.model}} is obtained from
##'   \code{\link{createSTmodel}}.
##' @param loc.ind A vector indicating which location each row in \code{F}
##'   corresponds to, usually \cr \code{\link{mesa.model}$obs$idx}.
##' @param n.loc Number of locations.
##' @return Returns the expanded F, a \code{dim(F)[1]}-by-\code{n.loc*dim(F)[2]}
##'   matrix (as a sparse-Matrix)
##' 
##' @example Rd_examples/Ex_expandF.R
##' 
##' @author Johan Lindstrom and Adam Szpiro
##' @family temporal trend functions
##' @export
expandF <- function(F, loc.ind, n.loc=max(loc.ind)){
  ##call cpp-code; internal error checking
  return( expandF_cpp(F, as.integer(loc.ind), as.integer(n.loc)) )
}##function expandF
