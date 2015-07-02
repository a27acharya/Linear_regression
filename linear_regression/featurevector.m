function[featureV]= featurevector(Xtrain, p)
%% enter the Value of P
% feature vetor will be [1 x x^2 ... x^p-1];

m = length(Xtrain);
PowerP = zeros(m,p);
for j = 1 : m
for  i= 1 : p
    PowerP(j,i) = Xtrain(j,1).^(i-1);
end

featureV = PowerP;

end