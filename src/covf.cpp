#include "covf.h"
#include <cmath> //std::pow, std::cyl_bessel_k

//std functions used
using std::cyl_bessel_k;
using std::pair;
using std::pow;
using std::size_t;
using std::swap;
using std::tgamma;
using std::unique_ptr;

//Eigen functions used
using Eigen::ArrayXd;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::VectorXd;
using Eigen::VectorXi;

//define a global constant for use in finite differences
const static double h = 1e-5;


/* Define a static const map that contains the string->covf mapping */
static const std::map<std::string, std::function<std::unique_ptr<covf_functor>()> > covfNamesMap = {
  {"exp", [](){return unique_ptr<covf_functor>(new covf_exp_functor);} },
  {"exponential", [](){return unique_ptr<covf_functor>(new covf_exp_functor);} },
  {"exp2", [](){return unique_ptr<covf_functor>(new covf_exp2_functor);} },
  {"exponential2", [](){return unique_ptr<covf_functor>(new covf_exp2_functor);} },
  {"gaussian", [](){return unique_ptr<covf_functor>(new covf_exp2_functor);} },
  {"cubic", [](){return unique_ptr<covf_functor>(new covf_cubic_functor);} },
  {"spherical", [](){return unique_ptr<covf_functor>(new covf_spherical_functor);} },
  {"matern", [](){return unique_ptr<covf_functor>(new covf_matern_functor);} },
  {"cauchy", [](){return unique_ptr<covf_functor>(new covf_cauchy_functor);} },
  {"iid", [](){return unique_ptr<covf_functor>(new covf_iid_functor);} }
};


/*Function used to select covariance function
 * Takes a string as input and returns a unique ptr to the selected class (or nullptr)
 */
std::unique_ptr<covf_functor> select_covf(const std::string& name){
  const auto i = covfNamesMap.find(name);
  if( i==covfNamesMap.end() ){
    return nullptr;
  }
  return i->second();
}//select_covf

std::vector<std::string> list_covf(){
  std::vector<std::string> res;
  res.reserve( covfNamesMap.size() );
  for(const auto& i : covfNamesMap){
    res.push_back(i.first);
  }
  return res;
}//list_covf


                    
//function that returns the status of what to differentiate.
//diff = <0,0> Function it self
//diff = <i,0> First derivative wrt i
//diff = <i,j> Second derivative wrt i,j. These are ordered i>=j
std::pair<std::size_t, std::size_t> covf_functor::diff_(const Eigen::Ref<const Eigen::VectorXi>& diff) const{
  //start with a 0,0 pair indicating no diff.
  pair<size_t, size_t> res(0,0);
  
  if( diff.size()==1 ){
    res.first = static_cast<size_t>( diff(0) );
  }else if( diff.size()>1 ){
    res.first = static_cast<size_t>( diff(0) );
    res.second = static_cast<size_t>( diff(1) );
  }
  
  //sanity check
  if( res.first > nPars() ){ res.first=0; }
  if( res.second > nPars() ){ res.second=0; }
  
  //order
  if( res.first < res.second ){
    swap(res.first, res.second);
  }
  
  //return final diff ordering.
  return res;
}//covf_functor::diff_

//internal check if parameters are of the right size
bool covf_functor::check(Eigen::Index n) const{
  return n==nPars();
}
void covf_functor::checkRcpp(Eigen::Index n) const{
  if( !check(n) ){
    Rcpp::stop("Expected %u parameters", nPars());
  }
}

//alternative operator taking only pars and dist matrix. Sets diff=0 and calls the three parameter version of itself.<
Eigen::MatrixXd covf_functor::operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                                         const Eigen::Ref<const Eigen::MatrixXd>& dist){
  return this->operator()(par, dist, VectorXi::Zero(2));
}

/* Virtual specialisation for 0 parameters */
unsigned int covf_functor0::nPars() const {
  return 0;
}
std::vector< std::string > covf_functor0::namesPars() const {
  return std::vector< std::string >(0);
}

/* Virtual specialisation for 2 parameters (range, sill) */
unsigned int covf_functor2::nPars() const {
  return 2;
}
std::vector< std::string > covf_functor2::namesPars() const {
  return std::vector< std::string >({"range", "sill"});
}

/* Virtual specialisation for 3 parameters (range, sill, shape) */
unsigned int covf_functor3::nPars() const {
  return 3;
}
std::vector< std::string > covf_functor3::namesPars() const {
  return std::vector< std::string >({"range", "sill", "shape"});
}

/*Internal functions for computing exponential covariances*/
//compute d*exp(d)
double r_exp_diff_r(double d){ return d*exp(d); }
//compute d*(d+2)exp(d)
double r_exp_diff_rr(double d){ return d*(d+2)*exp(d); }

/* Specialisations for exponential covariance function */
Eigen::MatrixXd covf_exp_functor::operator()(const Eigen::Ref<const Eigen::VectorXd>& par,
                                             const Eigen::Ref<const Eigen::MatrixXd>& dist,
                                             const Eigen::Ref<const Eigen::VectorXi>& diff){
  //check number of parameters
  checkRcpp( par.size() );

  //fix differentiation flags
  auto diff_pair = diff_(diff);
  
  //extract parameters
  double range = par(0);
  double sill = par(1);
  
  //in the following casts to array are used to obtain elementwise operations.
  //compute -d/range
  MatrixXd res = -dist/range;
  
  //compute either covariance or first or second derivatives
  if( diff_pair.first==0 ){
    res = sill * res.array().exp();
  }else if( diff_pair.second==0 ){
    if( diff_pair.first==1 ){
      //f'_range(d)
      res = -sill/range * res.array().unaryExpr( &r_exp_diff_r );
    }else{
      //f'_sill(d)
      res = res.array().exp();
    }
  }else{
    if( diff_pair.first==1 && diff_pair.second==1 ){
      //f''_range(d)
      res = sill/pow(range,2) * res.array().unaryExpr( &r_exp_diff_rr );
    }else if( diff_pair.first==2 && diff_pair.second==1 ){
      //f''_range,sill(d)
      res = -1.0/range * res.array().unaryExpr( &r_exp_diff_r );
    }else{
      //f''_sill(d)
      res.setZero(dist.rows(),dist.cols());
    }
  }
  return res;
}//covf_exp_functor::operator()

/*Internal functions for computing double exponential covariances*/
//compute d*(d+1.5)exp(d)
double r_exp2_diff_rr(double d){ return d*(d+1.5)*exp(d); }

/* Specialisations for double exponential/Gaussian covariance function */
Eigen::MatrixXd covf_exp2_functor::operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                                              const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                              const Eigen::Ref<const Eigen::VectorXi>& diff){
  //check number of parameters
  checkRcpp( par.size() );
  
  //fix differentiation flags
  auto diff_pair = diff_(diff);

  //extract parameters
  double range = par(0);
  double sill = par(1);
  
  //in the following casts to array are used to obtain elementwise operations.
  //compute -d^2/range^2
  MatrixXd res = -dist.array().square() / pow(range,2);

  //compute either covariance or first or second derivatives
  if( diff_pair.first==0 ){
    res = sill * res.array().exp();
  }else if( diff_pair.second==0 ){
    if( diff_pair.first==1 ){
      //f'_range(d)
      res = -2.0*sill/range * res.array().unaryExpr( &r_exp_diff_r );
    }else{
      //f'_sill(d)
      res = res.array().exp();
    }
  }else{
    if( diff_pair.first==1 && diff_pair.second==1 ){
      //f''_range(d)
      res = 4.0*sill/pow(range,2) * res.array().unaryExpr( &r_exp2_diff_rr );
    }else if( diff_pair.first==2 && diff_pair.second==1 ){
      //f''_range,sill(d)
      res = -2.0/range * res.array().unaryExpr( &r_exp_diff_r );
    }else{
      //f''_sill(d)
      res.setZero(dist.rows(),dist.cols());
    }
  }
  return res;
}//covf_exp2_functor::operator()


/*Internal functions for computing cubic covariances*/
//compute 1-7*d^2 + 8.75*d^3 -3.5*d^5+0.75*d^ = (d-1)^4 * (.75*d^3+3*d^2+4*d+1)
double r_cubic(double d){ 
  return pow(d-1.0,4) * (0.75*pow(d,3) + 3.0*pow(d,2) + 4.0*d + 1.0);
}
//compute -7 * d^2 * (d-1)^3 * (0.75*d^2+2.25*d+2)/r
double r_cubic_diff_r(double d){ 
  return -7.0 * pow(d,2) * pow(d-1.0,3) * (0.75*pow(d,2) + 2.25*d + 2.0);
}
//compute 21 * d^2 * (d-1)^2 * (2*d^3+4*d^2+d-2)
double r_cubic_diff_rr(double d){ 
  return 21.0 * pow(d,2) * pow(d-1.0,2) * (2.0*pow(d,3) + 4.0*pow(d,2) + d - 2.0);
}

/* Specialisations for cubic covariance function */
Eigen::MatrixXd covf_cubic_functor::operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                                               const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                               const Eigen::Ref<const Eigen::VectorXi>& diff){
  //check number of parameters
  checkRcpp( par.size() );
  
  //fix differentiation flags
  auto diff_pair = diff_(diff);
  
  //extract parameters
  double range = par(0);
  double sill = par(1);
  
  //in the following casts to array are used to obtain elementwise operations.
  //compute min(d/r,1)
  MatrixXd res = (dist/range).array().cwiseMin(1);
  
  //compute either covariance or first or second derivatives
  if( diff_pair.first==0 ){
    res = sill * res.array().unaryExpr( &r_cubic );
  }else if( diff_pair.second==0 ){
    if( diff_pair.first==1 ){
      //f'_range(d)
      res = sill/range * res.array().unaryExpr( &r_cubic_diff_r );
    }else{
      //f'_sill(d)
      res = res.array().unaryExpr( &r_cubic );
    }
  }else{
    if( diff_pair.first==1 && diff_pair.second==1 ){
      //f''_range(d)
      res = sill/pow(range,2) * res.array().unaryExpr( &r_cubic_diff_rr );
    }else if( diff_pair.first==2 && diff_pair.second==1 ){
      //f''_range,sill(d)
      res = 1.0/range * res.array().unaryExpr( &r_cubic_diff_r );
    }else{
      //f''_sill(d)
      res.setZero(dist.rows(),dist.cols());
    }
  }
  return res;
}//covf_cubic_functor::operator()

/*Internal functions for computing spherical covariances*/
//compute 1-1.5*d+0.5*d^3 = 0.5 * (d-1)^2 * (d+2)
double r_spherical(double d){ return 0.5 * pow(d-1.0,2) * (d+2.0); }
//compute -1.5 * d * (d^2-1)
double r_spherical_diff_r(double d){ return -1.5 * d * (pow(d,2)-1.0); }
//compute 3 * d * (2*d^2-1)
double r_spherical_diff_rr(double d){
  if(d<1){ 
    return 3.0 * d * (2.0*pow(d,2) - 1.0); 
  }
  return 0;
}

/* Specialisations for spherical covariance function */
Eigen::MatrixXd covf_spherical_functor::operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                                                   const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                                   const Eigen::Ref<const Eigen::VectorXi>& diff){
  //check number of parameters
  checkRcpp( par.size() );
  
  //fix differentiation flags
  auto diff_pair = diff_(diff);
  
  //extract parameters
  double range = par(0);
  double sill = par(1);
  
  //in the following casts to array are used to obtain elementwise operations.
  //compute min(d/r, 1)
  MatrixXd res = (dist/range).array().cwiseMin(1);
  
  //compute either covariance or first or second derivatives
  if( diff_pair.first==0 ){
    res = sill * res.array().unaryExpr( &r_spherical );
  }else if( diff_pair.second==0 ){
    if( diff_pair.first==1 ){
      //f'_range(d)
      res = sill/range * res.array().unaryExpr( &r_spherical_diff_r );
    }else{
      //f'_sill(d)
      res = res.array().unaryExpr( &r_spherical );
    }
  }else{
    if( diff_pair.first==1 && diff_pair.second==1 ){
      //f''_range(d)
      res = sill/pow(range,2) * res.array().unaryExpr( &r_spherical_diff_rr );
    }else if( diff_pair.first==2 && diff_pair.second==1 ){
      //f''_range,sill(d)
      res = 1.0/range * res.array().unaryExpr( &r_spherical_diff_r );
    }else{
      //f''_sill(d)
      res.setZero(dist.rows(),dist.cols());
    }
  }
  return res;
}//spherical

/*Internal functions for computing cauchy covariances*/
//compute (1+d)^-nu (assuming d has already been squared)
double r_cauchy(double d, double nu){ 
  return pow(1.0+d,-nu);
}
//compute 2 * nu * d * (d+1)^(-nu-1)
double r_cauchy_diff_r(double d, double nu){ 
  return 2.0 * nu * d * pow(1.0+d,-nu-1.0); 
}
//compute pow(-log(d+1),order) * (d+1)^(-nu)
double r_cauchy_diff_nu(double d, double nu, unsigned int order=1){ 
  return pow(-log(d+1),order) * pow(1.0+d,-nu); 
}
//compute 4 * nu * d * (d+1)^(-nu-2) * ((nu-0.5)*d - 1.5)
double r_cauchy_diff_rr(double d, double nu){ 
  return 4.0 * nu * d * pow(1.0+d,-nu-2.0) * ((nu-0.5)*d - 1.5); 
}
//compute -2 * d * (d+1)^(-nu-1) * (nu*ln(d+1)-1)
double r_cauchy_diff_rnu(double d, double nu){ 
  return -2.0 * d * pow(1.0+d,-nu-1) * (nu*log(1.0+d)-1.0); 
}

/* Specialisations for cauchy covariance function */
Eigen::MatrixXd covf_cauchy_functor::operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                                                const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                                const Eigen::Ref<const Eigen::VectorXi>& diff){
  //check number of parameters
  checkRcpp( par.size() );
  
  //fix differentiation flags
  auto diff_pair = diff_(diff);
  
  //extract parameters
  double range = par(0);
  double sill = par(1);
  double shape = par(2);
  
  //in the following casts to array are used to obtain elementwise operations.
  //compute (d/r)^2
  MatrixXd res = dist.array().square()/pow(range,2);
  
  //compute either covariance or first or second derivatives
  if( diff_pair.first==0 ){
    res = sill * res.array().unaryExpr( [shape](double x){ return r_cauchy(x,shape); } );
  }else if( diff_pair.second==0 ){
    if( diff_pair.first==1 ){
      //f'_range(d)
      res = sill/range * res.array().unaryExpr( [shape](double x){ 
        return r_cauchy_diff_r(x,shape); } );
    }else if( diff_pair.first==2 ){
      //f'_sill(d)
      res = res.array().unaryExpr( [shape](double x){ return r_cauchy(x,shape); } );
    }else{
      //f'_shape(d)
      res = sill * res.array().unaryExpr( [shape](double x){ 
        return r_cauchy_diff_nu(x,shape,1); } );
    }
  }else{
    if( diff_pair.first==1 && diff_pair.second==1 ){
      //f''_range(d)
      res = sill/pow(range,2) * res.array().unaryExpr( [shape](double x){ 
        return r_cauchy_diff_rr(x,shape); } );
    }else if( diff_pair.first==2 && diff_pair.second==1 ){
      //f''_range,sill(d)
      res = 1.0/range * res.array().unaryExpr( [shape](double x){ 
        return r_cauchy_diff_r(x,shape); } );
    }else if( diff_pair.first==3 && diff_pair.second==1 ){
      //f''_range,shape(d)
      res = sill/range * res.array().unaryExpr( [shape](double x){ 
        return r_cauchy_diff_rnu(x,shape); } );
    }else if( diff_pair.first==2 && diff_pair.second==2 ){
      //f''_sill(d), zero second derivative zero
      res.setZero(dist.rows(),dist.cols());
    }else if( diff_pair.first==3 && diff_pair.second==2 ){
      //f''_sill,shape(d)
      res = res.array().unaryExpr( [shape](double x){ 
        return r_cauchy_diff_nu(x,shape,1); } );
    }else{
      //f''_shape,shape(d)
      res = sill * res.array().unaryExpr( [shape](double x){ 
        return r_cauchy_diff_nu(x,shape,2); } );
    }
  }
  return res;
}//covf_cauchy_functor::operator()

/*Internal functions for computing Matern covariances*/
//compute D^nu * K_nu(D)
double r_matern(double d, double nu){
  return pow(d,nu)*cyl_bessel_k(nu,d);
}
//compute D^(nu+1) * K_(nu-1)(D)
double r_matern_diff_r(double d, double nu){
  return pow(d,nu+1) * cyl_bessel_k(abs(nu-1),d);
}
//compute D^(nu+1) * (D*K_(nu-2)(D) - 3*K_(nu-1)(D))
double r_matern_diff_rr(double d, double nu){
  return pow(d,nu+1) * ( d*cyl_bessel_k(abs(nu-2),d) - 3*cyl_bessel_k(abs(nu-1),d) );
}
//compute derivative wrt nu
double r_matern_diff_nu(double d, double nu, double digamma_nu){
  double bessel_D = cyl_bessel_k(nu,d);
  double diff_bessel = (cyl_bessel_k(nu+h,d)-bessel_D)/h;
  return pow(d,nu) * ( diff_bessel + bessel_D*(log(d)-log(2)-digamma_nu) -
             d/(2.0*nu)*cyl_bessel_k(abs(nu-1),d) );
}

/* Specialisations for matern covariance function */
Eigen::MatrixXd covf_matern_functor::operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                                                const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                                const Eigen::Ref<const Eigen::VectorXi>& diff){
  //check number of parameters
  checkRcpp( par.size() );

  //fix differentiation flags
  auto diff_pair = diff_(diff);

  //extract parameters
  double range = par(0);
  double sill = par(1);
  double shape = par(2);
  
  //precompute common constants
  double scaled_range = sqrt(8.0*shape)/range;
  double norm_const = 1.0 / ( tgamma(shape)*pow(2.0,shape-1.0) );
  //digamma function computed using finite differences
  double digamma_shape = (lgamma(shape+h)-lgamma(shape)) / h;
  
  //compute d*scaled_range
  MatrixXd res = dist*scaled_range;

  //in the following casts to array are used to obtain elementwise operations.
  if( diff_pair.first==0 ){
    res = sill*norm_const * res.array().unaryExpr( [shape](double x){
      return r_matern(x,shape); } );
  }else if( diff_pair.second==0 ){
    if( diff_pair.first==1 ){
      //f'_range(d)
      res = sill*norm_const/range * res.array().unaryExpr( [shape](double x){
        return r_matern_diff_r(x,shape); } );
    }else if( diff_pair.first==2 ){
      //f'_sill(d)
      res = norm_const * res.array().unaryExpr( [shape](double x){
        return r_matern(x,shape); } );
    }else{
      //f'_shape(d)
      res = sill*norm_const * res.array().unaryExpr( [shape,digamma_shape](double x){
        return r_matern_diff_nu(x,shape,digamma_shape); } );
    }
  }else{
    if( diff_pair.first==1 && diff_pair.second==1 ){
      //f''_range(d)
      res = sill*norm_const/(range*range) * res.array().unaryExpr( [shape](double x){
        return r_matern_diff_rr(x,shape); } );
    }else if( diff_pair.first==2 && diff_pair.second==1 ){
      //f''_range,sill(d)
      res = norm_const/range * res.array().unaryExpr( [shape](double x){
        return r_matern_diff_r(x,shape); } );
    }else if( diff_pair.first==3 && diff_pair.second==1 ){
      //f''_range,shape(d)
      Rcpp::stop("Analytical derivatives not available for Matern covariance.");
    }else if( diff_pair.first==2 && diff_pair.second==2 ){
      //f''_sill(d), zero second derivative zero
      res.setZero(dist.rows(),dist.cols());
    }else if( diff_pair.first==3 && diff_pair.second==2 ){
      //f''_sill,shape(d)
      res = norm_const * res.array().unaryExpr( [shape,digamma_shape](double x){
        return r_matern_diff_nu(x,shape,digamma_shape); } );
    }else{
      //f''_shape,shape(d)
      Rcpp::stop("Analytical derivatives not available for Matern covariance.");
    }
  }
  return res;
}//covf_matern_functor::operator()

/* Specialisations for iid covariance function */
Eigen::MatrixXd covf_iid_functor::operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                                           const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                           const Eigen::Ref<const Eigen::VectorXi>&){
  //check number of parameters
  checkRcpp( par.size() );
  
  //create and return empty matrix (nugget is added later)
  return MatrixXd::Zero(dist.rows(), dist.cols());
}//covf_iid_functor::operator()
