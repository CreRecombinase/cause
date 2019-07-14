
#include <algorithm>
#include <execution>
#include <RcppParallel.h>
#include <Rcpp.h>
using namespace Rcpp;
//using namespace tbb;

//RcppParallel::mutex m;


bool do_parallel;
int itc;


template<typename T,typename F>
void wrap_parallel_for (T block_obj, F && fun_obj){
  if(!do_parallel){
    #ifdef DEBUG
    if(itc++==0){

      Rcpp::Rcerr<<"running serial for-loop"<<std::endl;

    }
    #endif
    fun_obj(block_obj);
  }else{
#ifdef DEBUG
    if(itc++==0){
      Rcpp::Rcerr<<"running parallel for loop"<<std::endl;
    }
#endif
    // #ifdef __RCPP_PARALLEL__
    //     parallel_for(block_obj,fun_obj);
    // #else
    std::vector<size_t>	rs(block_obj.size());
    std::iota(rs.begin(),rs.end(),block_obj.begin());
    std::for_each(std::execution::par,rs.begin(),rs.end(),fun_obj);
    //    fun_obj(block_obj);
    //#endif

  }
}


using namespace RcppParallel;

//[[Rcpp::export]]
void reset_itc(){
  itc=0;
}

//[[Rcpp::export]]
bool set_parallel(bool parallel=false){
  do_parallel=parallel;
  reset_itc();
  return(do_parallel);
}

//[[Rcpp::export]]
bool get_parallel(){
  return(do_parallel);
}

// [[Rcpp::plugins(cpp11)]]

// likelihood of (\hat{\beta_1j}, \hat{\beta_2j}) given
// rho, gamma, eta, q, sigma_1j, sigma_2j, s_1j, s_2j
// we assume alpha_kj ~ N(0, sigma_kj)
//' @title Calculate likelihood of (beta_hat_1[j], beta_hat_2[j])) given
//' rho, b, sigma_1j, sigma_2j, s_1j, s_2j where
//' alpha_kj ~ N(0, sigma_kj)
//'@export
// [[Rcpp::export]]
double loglik_ij(const double rho, const double g, const double gp, const double q,
                const double  sigma1, const double  sigma2,
                const double b1, const double b2, const double s1, const double s2){

  //Likelihood if SNP acts directly on trait 1
  double v11_1 = std::pow(sigma1,2) + std::pow(s1, 2) ;
  double v12_1 = g*std::pow(sigma1,2) +  rho*s1*s2 ;
  double v22_1 = std::pow(sigma2,2) + std::pow(g*sigma1,2) +  std::pow(s2, 2) ;
  double m2_g1_1 = b1*v12_1/v11_1;
  double v2_g1_1 = v22_1 - std::pow(v12_1,2)/v11_1;


  double lik1_ij = log(1-q) + Rf_dnorm4(b1, 0, sqrt(v11_1), 1) +
    Rf_dnorm4(b2, m2_g1_1, sqrt(v2_g1_1), 1);



  //Likelihood if through the shared factor
  double v11_2 = std::pow(sigma1,2) + std::pow(s1, 2) ;
  double v12_2 = (g + gp)*std::pow(sigma1,2) +  rho*s1*s2 ;
  double v22_2 = std::pow(sigma2,2) + std::pow((g+gp)*sigma1,2) +  std::pow(s2, 2) ;
  double m2_g1_2 = b1*v12_2/v11_2;
  double v2_g1_2 = v22_2 - std::pow(v12_2,2)/v11_2;


  double lik2_ij = log(q) + Rf_dnorm4(b1, 0, sqrt(v11_2), 1) +
    Rf_dnorm4(b2, m2_g1_2, sqrt(v2_g1_2), 1);
  double lik_ij = Rf_logspace_add(lik1_ij, lik2_ij);

  return lik_ij;
}


// likelihood of (\hat{\beta_1j}, \hat{\beta_2j}) given
// rho, b, gamma, U, pi
// U is given as two vectors, sigma1 and sigma2
//pi is mixture parameters
double loglik_i(const double rho, const double g, const double gp, const double q,
                          const RVector<double> sigma1,
                          const RVector<double> sigma2,
                          const RVector<double> pi,
                          double b1, double b2, double s1, double s2){
  int k = sigma1.length();
  std::vector<double> lik_i(k);
  for(size_t j = 0; j < k; j ++){
    lik_i[j] = log(pi[j]) +
      loglik_ij(rho, g, gp, q,
              sigma1[j], sigma2[j],
              b1,  b2, s1, s2);
  }
  double result = Rf_logspace_sum(&lik_i[0], k);
  return result;
}



class logitij_fc{
  const double rho;
  const double g;
  const double gp;
  const double q;
  const RVector<double> beta_hat_1;//(tbeta_hat_1);
  const RVector<double> beta_hat_2;//(tbeta_hat_2);
  const RVector<double> seb2; //(tseb2);
  const RVector<double> seb1; //(tseb1);
  const RVector<double> sigma1; //(tsigma1);
  const RVector<double> sigma2; //(tsigma2);
  const int K;
  const int p = beta_hat_2.size();

  mutable RMatrix<double> lik_mat;//(tlik_mat);
public:
  logitij_fc(double rho_, double g_, double gp_, double q_,
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
    std::fill(lik_mat.begin(), lik_mat.end(), -1);
  }
  void operator()(const tbb::blocked_range<size_t>& r)const {
    for(size_t i=r.begin(); i!=r.end(); ++i){
      for(size_t j=0; j<K; j++){
	lik_mat(i,j) = loglik_ij(rho, g, gp, q,
				 sigma1[j],sigma2[j],
				 beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
      }
    }
  }
  void operator()(const size_t i) const{
    for(size_t j=0; j<K; j++){
      lik_mat(i,j) = loglik_ij(rho, g, gp, q,
			       sigma1[j],sigma2[j],
			       beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
    }
  }

};



//' @title Calculate likelihood matrix -- version 7
//' @return A matrix that is p by K where p is the number of SNPs
//' and K is the grid size
//'@export
// [[Rcpp::export]]
NumericMatrix loglik_mat(double rho, double g, double gp, double q,
                        NumericVector tsigma1, NumericVector tsigma2,
                        NumericVector tbeta_hat_1,
                        NumericVector tbeta_hat_2,
                        NumericVector tseb1,
                        NumericVector tseb2){

  // RVector<double> beta_hat_1(tbeta_hat_1);
  // RVector<double> beta_hat_2(tbeta_hat_2);
  // RVector<double> seb2(tseb2);
  // RVector<double> seb1(tseb1);
  // RVector<double> sigma1(tsigma1);
  // RVector<double> sigma2(tsigma2);



  int K = tsigma1.size();
  int p = tbeta_hat_2.size();
  NumericMatrix tlik_mat(p, K);
  logitij_fc f_obj(rho,  g,  gp,  q,  tsigma1,  tsigma2,  tbeta_hat_1,  tbeta_hat_2, tseb1,  tseb2,  tlik_mat);

  //  RMatrix<double> lik_mat(tlik_mat);




  wrap_parallel_for(tbb::blocked_range<size_t>(0, p),f_obj);
		    // [&](const blocked_range<size_t>& r){
		    //  for(size_t i=r.begin(); i!=r.end(); ++i){
		    //    for(size_t j=0; j<K; j++){
		    // 	 lik_mat(i,j) = loglik_ij(rho, g, gp, q,
		    // 				  sigma1[j],sigma2[j],
		    // 				  beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
		    //    }
		    //  }});

  return tlik_mat;
}



class logiti_fc{
  const double rho;
  const double g;
  const double gp;
  const double q;
  const RVector<double> beta_hat_1;//(tbeta_hat_1);
  const RVector<double> beta_hat_2;//(tbeta_hat_2);
  const RVector<double> seb2; //(tseb2);
  const RVector<double> seb1; //(tseb1);
  const RVector<double> sigma1; //(tsigma1);
  const RVector<double> sigma2; //(tsigma2);
  const RVector<double> pi;
  const int K;


  const int p = beta_hat_2.size();

  mutable RVector<double> ll;//(tlik_mat);
public:
  logiti_fc(double rho_, double g_, double gp_, double q_,
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
  void operator()(const tbb::blocked_range<size_t>& r)const {
    for(size_t i=r.begin(); i!=r.end(); ++i){
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



//' @title Calculate likelihood -- version 7
//'@export
// [[Rcpp::export]]
double loglik(double rho, double g, double gp, double q,
                        NumericVector tsigma1,
                        NumericVector tsigma2,
                        NumericVector tpi,
                        NumericVector tbeta_hat_1,
                        NumericVector tbeta_hat_2,
                        NumericVector tseb1,
                        NumericVector tseb2){

  using namespace RcppParallel;

  // const RVector<double> sigma1(tsigma1);
  // const RVector<double> sigma2(tsigma2);
  // const RVector<double> pi(tpi);

  // const RVector<double> beta_hat_1(tbeta_hat_1);
  // const RVector<double> beta_hat_2(tbeta_hat_2);
  // const RVector<double> seb2(tseb2);
  // const RVector<double> seb1(tseb1);


  const int p = tbeta_hat_2.size();
  NumericVector tll(p);
  //  RVector<double> ll(tll);
  logiti_fc lfc( rho,  g,  gp,  q,
	     tsigma1,
	     tsigma2,
	     tpi,
	     tbeta_hat_1,
	     tbeta_hat_2,
	     tseb1,
	     tseb2,tll);


  wrap_parallel_for(tbb::blocked_range<size_t>(0, p),lfc);
               // [&](const blocked_range<size_t>& r){
               //   for(size_t i=r.begin(); i!=r.end(); ++i){

               //     ll[i] = loglik_i(rho, g, gp, q,
               //                     sigma1,sigma2,pi,
               //                     beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
               //   }});
  double result = std::accumulate(tll.begin(),tll.end(),0);
  return result;
}

//' @title Calculate likelihood matrix for LOO (SNPs by posterior samples)
//' @return A matrix that is p by L where p is the number of SNPs
//' and L is the number of posterior samples
//'@export
// [[Rcpp::export]]
NumericMatrix loglik_loo(NumericVector tg, NumericVector tgp, NumericVector tq,
                        double rho,
                        NumericVector tsigma1, NumericVector tsigma2,
                        NumericVector tpi,
                        NumericVector tbeta_hat_1,
                        NumericVector tbeta_hat_2,
                        NumericVector tseb1,
                        NumericVector tseb2){

  const RVector<double> g(tg);
  const RVector<double> gp(tgp);
  const RVector<double> q(tq);

  const RVector<double> sigma1(tsigma1);
  const RVector<double> sigma2(tsigma2);
  const RVector<double> pi(tpi);

  const RVector<double> beta_hat_1(tbeta_hat_1);
  const RVector<double> beta_hat_2(tbeta_hat_2);
  const RVector<double> seb2(tseb2);
  const RVector<double> seb1(tseb1);

  int L = g.size();
  int p = beta_hat_2.size();

  NumericMatrix tlik_mat(p, L);
  RMatrix<double> lik_mat(tlik_mat);

  std::fill(lik_mat.begin(), lik_mat.end(), -1);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, p),
		    [&](const tbb::blocked_range<size_t>& r){
                 for(size_t i=r.begin(); i!=r.end(); ++i){
                   for(size_t j=0; j<L; j++){
                     lik_mat(i,j) = loglik_i(rho, g[j], gp[j], q[j],
                                            sigma1,sigma2,pi,
                                            beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
                   }
                 }});

  return tlik_mat;
}


//Functions for prob_confounding

// likelihood of (\hat{\beta_1j}, \hat{\beta_2j}) given
// rho, gamma, gamma_prime, q, sigma_1j, sigma_2j, s_1j, s_2j, Z_j = 1
// we assume alpha_kj ~ N(0, sigma_kj)
//'@export
// [[Rcpp::export]]
double loglik_ij_Z1(const double rho, const double g, const double gp, const double q,
                const double  sigma1, const double  sigma2,
                const double b1, const double b2, const double s1, const double s2){

  //Likelihood if through the shared factor
  double v11_2 = std::pow(sigma1,2) + std::pow(s1, 2) ;
  double v12_2 = (g + gp)*std::pow(sigma1,2) +  rho*s1*s2 ;
  double v22_2 = std::pow(sigma2,2) + std::pow((g+gp)*sigma1,2) +  std::pow(s2, 2) ;
  double m2_g1_2 = b1*v12_2/v11_2;
  double v2_g1_2 = v22_2 - std::pow(v12_2,2)/v11_2;

  double lik2_ij = log(q) + Rf_dnorm4(b1, 0, sqrt(v11_2), 1) +
    Rf_dnorm4(b2, m2_g1_2, sqrt(v2_g1_2), 1);
  return lik2_ij;
}


// likelihood of (\hat{\beta_1j}, \hat{\beta_2j}) given
// rho, gamma, gamma_prime, q, sigma_1j, sigma_2j, s_1j, s_2j, Z_j = 0
// we assume alpha_kj ~ N(0, sigma_kj)
//'@export
// [[Rcpp::export]]
double loglik_ij_Z0(const double rho, const double g, const double gp, const double q,
                const double  sigma1, const double  sigma2,
                const double b1, const double b2, const double s1, const double s2){

  //Likelihood if SNP acts directly on trait 1
  double v11_1 = std::pow(sigma1,2) + std::pow(s1, 2) ;
  double v12_1 = g*std::pow(sigma1,2) +  rho*s1*s2 ;
  double v22_1 = std::pow(sigma2,2) + std::pow(g*sigma1,2) +  std::pow(s2, 2) ;
  double m2_g1_1 = b1*v12_1/v11_1;
  double v2_g1_1 = v22_1 - std::pow(v12_1,2)/v11_1;


  double lik1_ij = log(1-q) + Rf_dnorm4(b1, 0, sqrt(v11_1), 1) +
    Rf_dnorm4(b2, m2_g1_1, sqrt(v2_g1_1), 1);

  return lik1_ij;
}

// likelihood of (\hat{\beta_1i}, \hat{\beta_2i}) given
// rho, gamma, eta, U, pi, Z_i = 1
// U is given as two vectors, sigma1 and sigma2
//pi is mixture parameters
double loglik_i_Z1(const double rho, const double g, const double gp, const double q,
               const RVector<double> &sigma1,
               const RVector<double> &sigma2,
               const RVector<double> &pi,
               double b1, double b2, double s1, double s2){
  int k = sigma1.length();
  std::vector<double> lik_i(k);
  for(size_t i = 0; i < k; i ++){
    lik_i[i] = log(pi[i]) +
    loglik_ij_Z1(rho, g, gp, q,
            sigma1[i], sigma2[i],
            b1,  b2, s1, s2);
  }
  double result = Rf_logspace_sum(&lik_i[0], k);
  return result;
}

// likelihood of (\hat{\beta_1i}, \hat{\beta_2i}) given
// rho, gamma, eta, U, pi, Z_i = 1
// U is given as two vectors, sigma1 and sigma2
//pi is mixture parameters
double loglik_i_Z0(const double rho, const double g, const double gp, const double q,
                  const RVector<double> &sigma1,
                  const RVector<double> &sigma2,
                  const RVector<double> &pi,
                  double b1, double b2, double s1, double s2){
  int k = sigma1.length();
  std::vector<double> lik_i(k);
  for(size_t i = 0; i < k; i ++){
    lik_i[i] = log(pi[i]) +
      loglik_ij_Z0(rho, g, gp, q,
                  sigma1[i], sigma2[i],
                                   b1,  b2, s1, s2);
  }
  double result = Rf_logspace_sum(&lik_i[0], k);
  return result;
}

//' @title Calculate likelihood matrix for LOO (SNPs by posterior samples)
//' @return A matrix that is p by L where p is the number of SNPs
//' and L is the number of posterior samples
//'@export
// [[Rcpp::export]]
NumericMatrix loglik_samps_Z1(NumericVector tg, NumericVector tgp, NumericVector tq,
                            double rho,
                            NumericVector tsigma1, NumericVector tsigma2,
                            NumericVector tpi,
                            NumericVector tbeta_hat_1,
                            NumericVector tbeta_hat_2,
                            NumericVector tseb1,
                            NumericVector tseb2){

  const RVector<double> g(tg);
  const RVector<double> gp(tgp);
  const RVector<double> q(tq);

  const RVector<double> sigma1(tsigma1);
  const RVector<double> sigma2(tsigma2);
  const RVector<double> pi(tpi);

  const RVector<double> beta_hat_1(tbeta_hat_1);
  const RVector<double> beta_hat_2(tbeta_hat_2);
  const RVector<double> seb2(tseb2);
  const RVector<double> seb1(tseb1);

  int L = g.size();
  int p = beta_hat_2.size();

  NumericMatrix tlik_mat(p, L);
  RMatrix<double> lik_mat(tlik_mat);

  std::fill(lik_mat.begin(), lik_mat.end(), -1);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, p),
		    [&](const tbb::blocked_range<size_t>& r){
                 for(size_t i=r.begin(); i!=r.end(); ++i){
                   for(size_t j=0; j<L; j++){
                     lik_mat(i,j) = loglik_i_Z1(rho, g[j], gp[j], q[j],
                             sigma1,sigma2,pi,
                             beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
                   }
                 }});

  return tlik_mat;
}


//' @title Calculate likelihood matrix for LOO (SNPs by posterior samples)
//' @return A matrix that is p by L where p is the number of SNPs
//' and L is the number of posterior samples
//'@export
// [[Rcpp::export]]
NumericMatrix loglik_samps_Z0(NumericVector tg, NumericVector tgp, NumericVector tq,
                             double rho,
                             NumericVector tsigma1, NumericVector tsigma2,
                             NumericVector tpi,
                             NumericVector tbeta_hat_1,
                             NumericVector tbeta_hat_2,
                             NumericVector tseb1,
                             NumericVector tseb2){

  const RVector<double> g(tg);
  const RVector<double> gp(tgp);
  const RVector<double> q(tq);

  const RVector<double> sigma1(tsigma1);
  const RVector<double> sigma2(tsigma2);
  const RVector<double> pi(tpi);

  const RVector<double> beta_hat_1(tbeta_hat_1);
  const RVector<double> beta_hat_2(tbeta_hat_2);
  const RVector<double> seb2(tseb2);
  const RVector<double> seb1(tseb1);

  int L = g.size();
  int p = beta_hat_2.size();

  NumericMatrix tlik_mat(p, L);
  RMatrix<double> lik_mat(tlik_mat);

  std::fill(lik_mat.begin(), lik_mat.end(), -1);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, p),
		    [&](const tbb::blocked_range<size_t>& r){
                 for(size_t i=r.begin(); i!=r.end(); ++i){
                   for(size_t j=0; j<L; j++){
                     lik_mat(i,j) = loglik_i_Z0(rho, g[j], gp[j], q[j],
                             sigma1,sigma2,pi,
                             beta_hat_1[i], beta_hat_2[i], seb1[i], seb2[i]);
                   }
                 }});

  return tlik_mat;
}





