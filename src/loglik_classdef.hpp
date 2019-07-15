#pragma once

#include <RcppParallel.h>


double loglik_ij(const double rho, const double g, const double gp, const double q,
                const double  sigma1, const double  sigma2,
		 const double b1, const double b2, const double s1, const double s2);


struct LogLik_obj: public RcppParallel::Worker
{
  const double rho;
  const double g;
  const double gp;
  const double q;
  const RcppParallel::RVector<double> beta_hat_1;//(tbeta_hat_1);
  const RcppParallel::RVector<double> beta_hat_2;//(tbeta_hat_2);
  const RcppParallel::RVector<double> seb2; //(tseb2);
  const RcppParallel::RVector<double> seb1; //(tseb1);
  const RcppParallel::RVector<double> sigma1; //(tsigma1);
  const RcppParallel::RVector<double> sigma2; //(tsigma2);
  const int K;
  const int p = beta_hat_2.size();
  mutable RMatrix<double> lik_mat;//(tlik_mat);

  LogLik_obj(double rho_, double g_, double gp_, double q_,
                        NumericVector tsigma1, NumericVector tsigma2,
                        NumericVector tbeta_hat_1,
                        NumericVector tbeta_hat_2,
                        NumericVector tseb1,
	     NumericVector tseb2,
	     NumericMatrix tlik_mat):rho(rho_),
				     g(g_),
				     gp(gp_),
				     q(q_),
				     beta_hat_1(tbeta_hat_1),
				     beta_hat_2(tbeta_hat_2),
				     seb2(tseb2),
				     seb1(tseb1),
				     sigma1(tsigma1),
				     sigma2(tsigma2),
				     K(sigma1.size()),
				     p(beta_hat_2.size()),
				     lik_mat(tlik_mat)

  {

  }
  void operator()(size_t begin, size_t end){
    for(size_t i=begin; i!=end; ++i){
    for(size_t j=0; j<K; j++){
      lik_mat(i,j) = loglik_ij(rho, g, gp, q,
			       sigma1[j],sigma2[j],
			       beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
    }
    }
  }
};

class logiti_fc:public RcppParallel::Worker{
  const double rho;
  const double g;
  const double gp;
  const double q;
  const RcppParallel::RVector<double> beta_hat_1;//(tbeta_hat_1);
  const RcppParallel::RVector<double> beta_hat_2;//(tbeta_hat_2);
  const RcppParallel::RVector<double> seb2; //(tseb2);
  const RcppParallel::RVector<double> seb1; //(tseb1);
  const RcppParallel::RVector<double> sigma1; //(tsigma1);
  const RcppParallel::RVector<double> sigma2; //(tsigma2);
  const RcppParallel::RVector<double> pi;
  const int K;


  const int p = beta_hat_2.size();

  mutable RcppParallel::RVector<double> ll;//(tlik_mat);
public:
  logiti_fc(double rho_,
	    double g_,
	    double gp_,
	    double q_,
	    NumericVector tsigma1, NumericVector tsigma2,
	    NumericVector tpi,
	    NumericVector tbeta_hat_1,
	    NumericVector tbeta_hat_2,
	    NumericVector tseb1,
	    NumericVector tseb2,
	    NumericVector tll):rho(rho_),
			       g(g_),
			       gp(gp_),
			       q(q_),
			       beta_hat_1(tbeta_hat_1),
			       beta_hat_2(tbeta_hat_2),
			       seb2(tseb2),
			       seb1(tseb1),
			       sigma1(tsigma1),
			       sigma2(tsigma2),
			       pi(tpi),
			       K(sigma1.size()),
			       p(beta_hat_2.size()),
			       ll(tll)

  {
    std::fill(ll.begin(), ll.end(), -1);
  }
  void operator()(const size_t begin, const size_t end) {
    for(size_t i=begin; i!=end; ++i){
      ll[i] = loglik_i(rho, g, gp, q,
		       sigma1,sigma2,pi,
		       beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
    }
  }
    void operator()(const size_t i)const {
      ll[i] = loglik_i(rho, g, gp, q,
		       sigma1,sigma2,pi,
		       beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
    }
};
