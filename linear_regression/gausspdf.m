function p = gausspdf(X, mu, Sigma)
% Multivariate Gaussian distribution, pdf
d = size(Sigma, 2);
X = bsxfun(@minus, X, mu);
logp = -0.5*sum((X/(Sigma)).*X, 2) - (d/2)*log(2*pi) + 0.5*log(det(Sigma));
p = exp(logp);        
end