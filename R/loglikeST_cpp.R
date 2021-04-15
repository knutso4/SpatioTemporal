#################################################
## FILE CONTAINING THE LOGLIKELIHOOD FUNCTIONS ##
#################################################
##Functions in this file:
## loglikeST                EX:ok
## loglikeSTnaive           EX:with loglikeST
## loglikeSTinit            EX:with loglikeST
## loglikeST_internal       EX:with loglikeST
## loglikeSTnaive_internal  EX:with loglikeST

##' Computes the log-likelihood for the spatio-temporal model.  \code{loglikeST}
##' uses an optimized version of the log-likelihood, while \code{loglikeSTnaive}
##' uses the naive (slow) version and is included mainly for testing and speed
##' checks.
##' 
##' For multiple calls with the same \code{STmodel} (as in optimization or MCMC-runs) 
##' its recommend to use \code{loglikeSTinit} and followed by calls to 
##' \code{loglikeST_internal}, see example code.
##' 
##' @title Compute the Log-likelihood for the Spatio-Temporal Model
##' @param x Point at which to compute the log-likelihood, should be only
##'   \emph{log}-covariance parameters if \code{type=c("p","r")} and
##'   regression parameters followed by \emph{log}-covariance parameters if
##'   \code{type="f"}. If \code{x=NULL} the function acts as an alias for
##'   \code{\link{loglikeSTnames}} returning the expected names of the
##'   input parameters.
##' @param STmodel \code{STmodel} object with the model for which to compute
##'   the log-likelihood.
##' @param type A single character indicating the type of log-likelihood to
##'   compute. Valid options are "f", "p", and "r", for \emph{full},
##'   \emph{profile} or \emph{restricted maximum likelihood} (REML).
##' @param x.fixed Parameters to keep fixed, \code{NA} values in this vector is
##'   replaced by values from \code{x} and the result is used as \code{x}, ie. \cr
##'   \code{ x.fixed[ is.na(x.fixed) ] <- x} \cr \code{ x <- x.fixed }.
##' 
##' @return Returns the log-likelihood of the spatio temporal model. 
##' 
##' @section Warning: \code{loglikeSTnaive} may take long to run. However for
##'   some problems with many locations and short time series
##'   \code{loglikeSTnaive} could be faster than \code{loglikeST}.
##' 
##' @example Rd_examples/Ex_loglikeST.R
##'
##' @author Johan Lindstrom
##' 
##' @family STmodel functions
##' @family likelihood functions
##' @family estimation functions
##' @export
loglikeST <- function(x=NULL, STmodel, type=c("p","r","f"), x.fixed=NULL){
  ##check class belonging
  stCheckClass(STmodel, "STmodel", name="STmodel")
  ##first ensure that type is ok
  type <- match.arg(type)
  ##check if type is valid and if x.fixed should be expanded
  x <- stCheckLoglikeIn(x, x.fixed, type)
  
  ##if x is null or contains NA
  if( is.null(x) || any(is.na(x)) ){
    ##return the expected variable names
    return( loglikeSTnames(STmodel, all=(type=="f")) )
  }
  
  ##else, calculate loglikelihood
  STinit <- loglikeSTinit(STmodel, type, x.fixed)
  loglikeST_internal(x, STinit)
}

###############################################
## Log-likelihood using the full formulation ##
###############################################
##' @rdname loglikeST
##' @export
loglikeSTnaive <- function(x=NULL, STmodel, type=c("p","r","f"), x.fixed=NULL){
  ##check class belonging
  stCheckClass(STmodel, "STmodel", name="STmodel")
  ##first ensure that type is ok
  type <- match.arg(type)
  ##check if type is valid and if x.fixed should be expanded
  x <- stCheckLoglikeIn(x, x.fixed, type)
  
  ##if x is null or contains NA
  if( is.null(x) || any(is.na(x)) ){
    ##return the expected variable names
    return( loglikeSTnames(STmodel, all=(type=="f")) )
  }
  
  ##else, calculate loglikelihood
  STinit <- loglikeSTinit(STmodel, type, x.fixed)
  loglikeSTnaive_internal(x, STinit)
}

##########################################################
## Log-likelihood init code for using internal cpp-code ##
##########################################################
##' @rdname loglikeST
##' @export
loglikeSTinit <- function(STmodel, type, x.fixed=NULL){
  ##check class belonging
  stCheckClass(STmodel, "STmodel", name="STmodel")

  ##first figure out a bunch of dimensions
  dim <- loglikeSTdim(STmodel)
  
  ##set x.fixed as a vector of NA if missing
  nparam <- ifelse(type=='f', dim$nparam, dim$nparam.cov)
  if( is.null(x.fixed) ){
    x.fixed <- rep(NA, nparam)
  }else if( length(x.fixed)!=nparam ){
    ##sanity check of size of x.fixed
    stop("x.fixed should be NULL or contain ", nparam, ' elements')
  }
  
  ##Create return list of values
  ret <- list(x_fixed=as.double(x.fixed), type=type, Y_in=STmodel$obs$obs,
              F=expandF(STmodel$F, STmodel$obs$idx, n.loc=dim$n.obs))
  ##precompute [M F*X], using that cbind(NULL,F*X) = F*X for the case of no M-matrix
  ret$M_FX <- cbind(STmodel[['ST']], as.matrix(ret$F %*% bdiag(STmodel$LUR)))

  ##type of covariance for sigmaB
  ret$type_B <- STmodel$cov.beta$covf
  ##type of covariance for sigmaNu
  ret$type_nu <- STmodel$cov.nu$covf
  ##description of nugget for nu-field
  ret$nugget_matrix <- STmodel$cov.nu$nugget.matrix
  
  ##Number of parameters for SigmaB and SigmaNu
  ##Parameter order: (beta1, nugget_beta1, ..., SigmaNu, SigmaNu_nugget)
  ret$n_cov_pars <- as.integer(c(rbind(dim$npars.beta.covf, dim$npars.beta.tot-dim$npars.beta.covf), 
                                 dim$npars.nu.covf, dim$npars.nu.tot-dim$npars.nu.covf))
  
  ##Distance matrices
  ret$dist_B <- STmodel$D.beta
  ret$dist_nu <- STmodel$D.nu

  ##Structure of SigmaNu
  ret$blocks <- as.integer(STmodel$nt)
  ret$ind <- as.integer(STmodel$obs$idx)
  
  ##return the list of precomputed values.
  return(ret)
}

###################################
## Wrapers for internal cpp-code ##
###################################
##' @rdname loglikeST
##' @param STinit Output from call to \code{loglikeSTinit}, see details.
##' @export
loglikeST_internal <- function(x, STinit){
  return( do.call(loglikeST_cpp ,c(list(x_in=x), STinit)) )
}

##' @rdname loglikeST
##' @export
loglikeSTnaive_internal <- function(x, STinit){
  return( do.call(loglikeSTnaive_cpp ,c(list(x_in=x), STinit)) )
}
