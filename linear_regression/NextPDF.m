function [postW,postMu,postSigma] = NextPDF(xtrain,ytrain,likelihoodPrecision,priorMu,priorSigma)

postSigma  = inv(inv(priorSigma) + (likelihoodPrecision*(xtrain'*xtrain)));
postMu = (postSigma)*((inv(priorSigma))*priorMu) + likelihoodPrecision*(postSigma)*(xtrain'*ytrain);
postW = @(W)gausspdf(W,postMu',postSigma);

end