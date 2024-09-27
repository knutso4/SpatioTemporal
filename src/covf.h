#ifndef COVF_H
#define COVF_H

#include <RcppEigen.h>
#include <memory>  //std::unique_ptr
#include <string>  //std::string
#include <utility> //std::pair
#include <vector>  //std::vector

/* Abstract class that works as interface for covariance functions */
class covf_functor{
protected:
  //function that returns the status of what to differentiate.
  std::pair<std::size_t, std::size_t> diff_(const Eigen::Ref<const Eigen::VectorXi>& diff) const;
  bool check(Eigen::Index n) const;
  void checkRcpp(Eigen::Index n) const;
  
public:
  //default virtual destructor
  virtual ~covf_functor() = default;
  
  //and define the interface functions.
  //Given parameters, distance and possibly a vector of differences should
  //return values of the covariance function, without nugget.
  virtual Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                                     const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                                     const Eigen::Ref<const Eigen::VectorXi>& diff) =0;
  //alternative operator taking only pars and dist matrix. Sets diff=0 and calls the function above.
  Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                             const Eigen::Ref<const Eigen::MatrixXd>& dist);
  
  //return the number of parameters
  virtual unsigned int nPars() const =0;
  //and names of the parameters
  virtual std::vector<std::string> namesPars() const =0;
};


/* Virtual specialisation for 0 parameters */
class covf_functor0 : public covf_functor{
public:
  //default virtual destructor
  virtual ~covf_functor0() = default;
  //return the number of parameters
  unsigned int nPars() const;
  //and names of the parameters
  std::vector< std::string > namesPars() const;
};

/* Virtual specialisation for 2 parameters (range, sill) */
class covf_functor2 : public covf_functor{
public:
  //default virtual destructor
  virtual ~covf_functor2() = default;
  //return the number of parameters
  unsigned int nPars() const;
  //and names of the parameters
  std::vector< std::string > namesPars() const;
};

/* Specialisation for 3 parameters (range, sill, shape) */
class covf_functor3 : public covf_functor{
public:
  //default virtual destructor
  virtual ~covf_functor3() = default;
  //return the number of parameters
  unsigned int nPars() const;
  //and names of the parameters
  std::vector< std::string > namesPars() const;
};

/* Function that returns a list of possible covariance functions*/
std::vector<std::string> list_covf();
  
/* Function used to select covariance function
 * Takes a string as input and returns a unique ptr to the selected class (or nullptr)
 */
std::unique_ptr<covf_functor> select_covf(const std::string& name);

/* COVARIANCE FUNCTIONS */
//Given parameters, distance and possibly a vector of differences should
//return values of the covariance function, without nugget.

/* Specialisations for exponential covariance function */
class covf_exp_functor : public covf_functor2{
  Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                             const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                             const Eigen::Ref<const Eigen::VectorXi>& diff);
};

/* Specialisations for double exponential/Gaussian covariance function */
class covf_exp2_functor : public covf_functor2{
  Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                             const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                             const Eigen::Ref<const Eigen::VectorXi>& diff);
};

/* Specialisations for cubic covariance function */
class covf_cubic_functor : public covf_functor2{
  Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                           const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                           const Eigen::Ref<const Eigen::VectorXi>& diff);
};

/* Specialisations for spherical covariance function */
class covf_spherical_functor : public covf_functor2{
  Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                           const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                           const Eigen::Ref<const Eigen::VectorXi>& diff);
};

/* Specialisations for cauchy covariance function */
class covf_cauchy_functor : public covf_functor3{
  Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                           const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                           const Eigen::Ref<const Eigen::VectorXi>& diff);
};

/* Specialisations for matern covariance function */
class covf_matern_functor : public covf_functor3{
  Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                           const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                           const Eigen::Ref<const Eigen::VectorXi>& diff);
};

/* Specialisations for iid covariance function */
class covf_iid_functor : public covf_functor0{
  Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::VectorXd>& par, 
                             const Eigen::Ref<const Eigen::MatrixXd>& dist, 
                             const Eigen::Ref<const Eigen::VectorXi>& diff);
};

#endif /* COVF_H */