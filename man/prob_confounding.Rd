% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prob_confounding.R
\name{prob_confounding}
\alias{prob_confounding}
\title{Posterior probability of association with confounder}
\usage{
prob_confounding(X, fit, nsamps = 1000, name_post = "")
}
\arguments{
\item{X}{object of class cause_data. This need not be the same data used to compute 'fit'.}

\item{fit}{object of class cause_post.}

\item{nsamps}{Number of samples to take from the posterior (see Details).}
}
\value{
A vector of probabilities corresponding to the rows of X
}
\description{
For a set of data, X, calculate the probability that each variant
acts through the confounder
}
\details{
Let x_i represent the the data in the ith row of 'X'. Let Z_i be the indicator that
variant i acts through the confounder. We are interested in computing P(Z_i = 1 | posteriors) where
posteriors refers to the posterior parameter distributions contained in  'fit'.
We compute l(x_i | posteriors, Z=1) and l(x_i | posteriors, Z_i = 0) where l denotes the likelihood.
We then comput P(Z_i = 1 | posteriors) = l(x_i | posteriors, Z_i = 1)/(l(x_i | posteriors, Z_i = 1) + l(x_i | posteriors, Z_i = 0))
}
