## crossDist      EX:ok

##' Computed the Euclidian distance matrix between to sets of points.
##'
##' @title Computed the Euclidian Distance Matrix
##' @param coord1,coord2 Matrices with the coordinates of locations, between
##'   which distances are to be computed.
##' @return A \code{dim(coord1)[1]}-by-\code{dim(coord2)[1]} distance matrix.
##' 
##' @example Rd_examples/Ex_crossDist.R
##' 
##' @author Johan Lindstrom
##' @family covariance functions
##' @family basic linear algebra
##' @export
crossDist <- function(coord1, coord2=coord1){
  if( missing(coord2) ){
    return( dist_Cpp(as.matrix(coord1)) )
  }else{
    return( crossDist_Cpp(as.matrix(coord1), as.matrix(coord2)) )
  }
}##crossDist <- function

