clear all;
close all;
clc;

%% Input data 
X = load('task1.csv');
[m,n] = size(X);
% training data set
Xtrain = X(1:10,1);
Ytrainold = X(1:10,2);
%validation data set
Xval = X(11:1000,1);
Yvalold = X(11:1000,2);

% cell for a Wmle as matrix dimensions will change with feature vector
Ctrain = cell(12,1);


%% For training data set
for p = 2 : 13
   
% call funcion feature vector to calulate FV for p
%note : for p = 2 we will get FV of [1x2] dimension
[featureV]= featurevector(Xtrain, p);
featureVtrain = featureV;

% calculate value of Wmle by following equation
%Wmle = inv((Xtrain)'*(Xtrain))*(Xtrain'*Ytrain);
Wmletrain = inv((featureVtrain'*featureVtrain))*(featureVtrain'*Ytrainold);
Xnew = featureVtrain'; 
Ctrain{p-1} = Wmletrain;

%Ctrain{1} = [2x1]matrix  Xtrain = [1 x1];
% mean square error = (1/n)* sum((Ygiven - Ypred).^2);
MSEtrain(1,p-1) =(1/10)*sum((Ytrainold - diag((repmat(Ctrain{p-1,1},1,10)')*(featureVtrain'))).^2);
Ytrainnew = diag(repmat(Ctrain{p-1,1}',m,1) * Xnew) ;

%%plot models output using obtained estimate of W
h(p)=figure(p-1);
plot(Xtrain(:,1),Ytrainnew(:,1),'b*');

xlabel(['when value of p for feature vector is  ' , num2str(p-1)]);
ylabel('y(x/WMean Square Error)');
title('models output using obtained estimate of W');
%    filename = sprintf('test_image_Task2_%d.png', p);
%     saveas(h(p),filename);
end


%% For validation data set

Cval = cell(12,1);
for p = 2 : 13
[featureV]= featurevector(Xval, p);
featureVval = featureV;
%Wmle = inv((Xtrain)'*(Xtrain))*(Xtrain'*Ytrain);
Wmleval = inv((featureVval'*featureVval))*(featureVval'*Yvalold);
Xnew = featureVval'; 

Cval{p-1} = Wmleval;

MSEval(1,p-1) =(1/990)*sum((Yvalold - diag((repmat(Cval{p-1,1},1,990)')*featureVval')).^2);
Yvalnew = diag(repmat(Cval{p-1}',m,1) * Xnew) ;

 %plot models output using obtained estimate of W) 
 %figure(p);
 %plot(Xval(:,1),Yvalnew(:,1),'b.');

end


%% plot MSE of training and validation data sets 
x=1:1:12;
F = figure(13);
plot(x,log(MSEtrain(1,:)),'r*');
hold on
plot(x,log(MSEval(1,:)),'b*');
legend('MSEtrain','MSEvalidation');
xlabel('value of p for feature vector')
ylabel('Mean Square Error')
title('MSE for training and validation set as feature vector changes');
%   filename = sprintf('test_image_Task2_%d.png', 13);
%    saveas(F,filename);

%% Regularization factor
% for different values of regularization factor plot MSE of training 
% and validation data set
% feature vector = [1 x . . x^5];
k=100;
%% for training data set
Ctrain5 = cell(k,1);
for u = 1 : k
% for given feature vector, p=6
[featuretrainV5]= featurevector(Xtrain, 8);
% Wmle for each value of regularization factor 
Wmletrain5 = inv((featuretrainV5'*featuretrainV5 + (u*eye(8))))*(featuretrainV5'*Ytrainold);
Ctrain5{u} = Wmletrain5;
% MSE for every value of regularization factor 
MSEtrain5(1,u) =(1/10)*sum((Ytrainold - diag((repmat(Ctrain5{u,1},1,10)')*(featuretrainV5'))).^2);
end

%% For validation data set
Cval5 = cell(k,1);
for u = 1 : k
% for given feature vector, p=6
[featurevalV5]= featurevector(Xval, 8);
% Wmle for each value of regularization factor 
Wmleval5 = inv((featurevalV5'*featurevalV5 + (u*eye(8))))*(featurevalV5'*Yvalold);
Cval5{u} = Wmleval5;
% MSE for every value of regularization factor 
MSEval5(1,u) =(1/990)*sum((Yvalold - diag((repmat(Cval5{u,1},1,990)')*(featurevalV5'))).^2);
end

%% Plot MSE of training and validation data sets 
x=1:1:k;
Fig = figure(14);
plot(x,log(MSEtrain5(1,:)),'r*');
hold on
plot(x,log(MSEval5(1,:)),'b*');
legend('MSEtrain','MSEvalidation');
xlabel('Regularization Factor (Mu) ')
ylabel('Mean Square Error')
title('MSE for P=7 for training and validation set as regularization factor changes');
%    filename = sprintf('test_image_Task2_%d.png', 14);
%    saveas(Fig,filename);
%%








%%