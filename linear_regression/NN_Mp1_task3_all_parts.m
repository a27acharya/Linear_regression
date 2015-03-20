clc;
close all;
clear all;

X = load('housing.data');
[m,n] = size(X);
range = zeros(m,1);
X2 = zeros(m,n);
range1 = zeros(m,1);
%% Normalize to [-1,1]:
 for j = 1 : n
     min2 = min(X(:,j));
     range1(j)= max(X(:,j)) - min2;
    for i = 1:m
     X2(i,j) = ((X(i,j) - min2) / (range1(j))- 0.5 ) * 2;
    end
 end
 
 %% Input data
 Xtrain = X2(1:306,1:13);
 Xval = X2(307:406,1:13);
 Xtest = X2(407:506,1:13);
 % output data
 Ytrainold = X2(1:306,14);
 Yvalold = X2(307:406,14);
 Ytestold = X2(407:506,14);
 
k = 100; 

%%RR Regression (part a)
%% For training data set

Ctrain = cell(k,1);
for u = 1 : k
% Wmle for each value of regularization factor 
Wmletrain = inv((Xtrain'*Xtrain + (u*eye(13))))*(Xtrain'*Ytrainold);
Ctrain{u} = Wmletrain;
% MSE for every value of regularization factor 
MSEtrainRR(1,u) =(1/306)*sum((Ytrainold - diag((repmat(Ctrain{u,1},1,306)')*(Xtrain'))).^2);
end

%% For validation data set
Cval = cell(k,1);
for u = 1 : k
% Wmle for each value of regularization factor 
Wmleval = inv((Xval'*Xval + (u*eye(13))))*(Xval'*Yvalold);
Cval{u} = Wmleval;
% MSE for every value of regularization factor 
MSEvalRR(1,u) =(1/100)*sum((Yvalold - diag((repmat(Cval{u,1},1,100)')*(Xval'))).^2);
end

%% MSE test RR
% for regularization factor =0 MSE is min for validation set 
% champion RR model will be weights corresponding to min MSEvalRR
% which we obtained for mu = 0.
% MSEvalRR{1,1} is optimal weights
% lets calulate the MSEtestRR for this chamipon model
for i = 1 : 100
MSEtestRR(1,i) =(1/100)*sum((Ytestold - diag((repmat(Cval{i,1},1,100)')*(Xtest'))).^2);
%%
end


%% Plot MSE of training and validation data sets
x=1:1:k;
Fig = figure(1);
 plot(x,log(MSEtrainRR(1,:)),'r*');
 hold on
plot(x,log(MSEvalRR(1,:)),'b*');
legend('MSEtrain','MSEvalidation');
xlabel('Regularization Factor (Mu) ')
ylabel('Mean Square Error')
title('MSE for validation set as regularization factor changes');

 

%%  RBF NN (part b)
[m,n] = size(Xval);
lambda = 100;
phi = cell(m,lambda);
phinew = cell(m,lambda);
array = zeros(m,1);
for k = 1 : lambda
for i = 1 : m 
       for j = 1 : m
   array(j,:) = exp(- k *sum (( Xval(i,:)- Xval(j,:)).^2));
       end 
       phi{i,k} = array';
end
end


%% For validation data set
CvalRBF = cell(lambda,1);
MSEvalRBF = zeros(1,lambda);
for u = 1 : lambda
% Wmle for each value of regularization factor 
phinew = vertcat(phi{:,u});
WmlevalRBF = inv((phinew'*phinew))*(phinew'*Yvalold);
CvalRBF{u} = WmlevalRBF;
% MSE for every value of regularization factor 
MSEvalRBF(1,u) =(1/100)*sum((Yvalold - diag((repmat(CvalRBF{u,1},1,100)')*(phinew'))).^2);
end

%% Plot MSE of training and validation data sets 
x=1:1:k;
Fig = figure(2);
 plot(x,log(MSEvalRBF(1,:)),'r*');
legend('MSEvalidation');
xlabel('Regularization Factor (Lambda) ')
ylabel('Mean Square Error for RBF model')
title('MSE for validation set as regularization factor changes');


%% For test data set
[m,n] = size(Xtest);
phitest = cell(m,1);
phinew = cell(m,lambda);
array = zeros(m,1);

for i = 1 : m 
       for j = 1 : m
   array(j,:) = exp(- 90 * sum (( Xtest(i,:)- Xtest(j,:)).^2));
       end 
       phitest{i,1} = array';
end

%% For test data set (part c)
% champion model is RBF NN
% from above evaluation we found out that MSEvalRBF is a best model
% for Lambda = 90, it gives the min MSE
% lets calculate the MSE for test set considering this value of Lambda
CtestRBF = cell(1);
MSEtestRBF = zeros(1,1);
% Wmle for regularization factor = 90 
phinewtest = vertcat(phitest{:,1});
% MSE test for test data set 
% weights obtained from champion model for lambda = 90 
%[MSEtestRBF2] = testRBFNN(Xtest,lambda,Ytestold,CvalRBF);
[MSEtestRBF] = testRBFNN(Xtest,90,Ytestold,CvalRBF);

% W = (1/13) * 1D; D = 13;
% MSE test without considering RR
Weighttest = (1/13) * ones(13,1);
MSEtest(1) = (1/100)*sum((Ytestold - diag((repmat(Weighttest,1,100)')*(Xtest'))).^2);
