#include "covf.h"
#include <memory>  //std::unique_ptr
#include <string>  //std::string
#include <vector>  //std::vector

// [[Rcpp::export]]
std::vector<std::string> parsCovFuns_cpp(const std::string& type){
  //first check which covariance function
  std::unique_ptr<covf_functor> covf = select_covf(type);
  if( covf==nullptr ){ 
    Rcpp::stop("Unknown covarince function: %s", type);
  }
  return covf->namesPars();
}//parsCovFuns_cpp

//' Returns a list of possible covariance function names
//'
//' Available covariance functions (\code{d} is the distance
//' between points):
//' \describe{
//'   \item{\code{exp,exponential}}{Exponential covariance:
//'     \deqn{\sigma^2 \exp(-d/\rho)}{sill * exp( -d/range )} }
//'   \item{\code{exp2,exponential2,gaussian}}{Gaussian/double
//'     exponential covariance:
//'     \deqn{\sigma^2 \exp(-(d/\rho)^2)}{sill * exp( -(d/range)^2 )} }
//'   \item{\code{cubic}}{Cubic covariance: 
//'     \deqn{\sigma^2 (1 - 7 (d/\rho)^2 + 8.75 (d/\rho)^3 -
//'           3.5 (d/\rho)^5 + 0.75 (d/\rho)^7)}{sill*(1 -
//'           7*(d/range)^2 + 8.75*(d/range)^3 - 3.5*(d/range)^5 +
//'           0.75*(d/range)^7)}
//'     if \eqn{d<\rho}{d<range}.}
//'   \item{\code{spherical}}{Spherical covariance: 
//'     \deqn{\sigma ^2(1 - 1.5(d/\rho) + 0.5 (d/\rho)^3)}{sill * (1 -
//'           1.5*(d/range) + 0.5*(d/range)^3)}
//'     if \eqn{d<\rho}{d<range}.}
//'   \item{\code{matern}}{Matern covariance: 
//'     \deqn{\frac{\sigma^2}{\Gamma(\nu) 2^{\nu-1}}
//'           \left(\frac{d\sqrt{8\nu}}{\rho}\right)^\nu
//'           K_\nu\left(\frac{d\sqrt{8\nu}}{\rho}\right)}{sill /
//'           (gamma(shape)*2^(shape-1)) *
//'           (d*sqrt(8*shape)/range)^shape *
//'           besselK( (d*sqrt(8*shape)/range), shape ) } }
//'   \item{\code{cauchy}}{Cauchy covariance: 
//'     \deqn{ \frac{\sigma^2}{(1 + (d/\rho)^2)^{\nu}}}{sill *
//'           (1 + (d/range)^2) ^ -shape} }
//'   \item{\code{iid}}{IID covariance, i.e. zero matrix 
//'     since nugget is added afterwards. \deqn{0}{0}}
//' }
//' 
//' @title Available covariance functions
//' @return Character vector with valid covariance function names.
//' 
//' @examples
//'   namesCovFuns()
//' 
//' @author Johan Lindstrom
//' @family covariance functions
//' @import Rcpp
//' @export
// [[Rcpp::export]]
std::vector<std::string> namesCovFuns(){
  return list_covf();
}//namesCovFuns

// [[Rcpp::export]]
Eigen::MatrixXd evalCovFuns_cpp(const std::string& type, 
                                     const Eigen::Map<Eigen::VectorXd>& pars, 
                                     const Eigen::Map<Eigen::MatrixXd>& d, 
                                     const Eigen::Map<Eigen::VectorXi>& diff){
  //first check which covariance function
  std::unique_ptr<covf_functor> covf = select_covf(type);
  if( covf==nullptr ){
    Rcpp::stop("Unknown covarince function: %s", type);
  }
  return covf->operator()(pars, d, diff);
}//evalCovFuns_cpp

  