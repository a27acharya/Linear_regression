clc;
close all;

%% Input data 
X = load('task1.csv');
[m,n] = size(X);
Xtrain = X(:,1);
Xtrain = [ones(m,1) Xtrain];
Ytrain = X(:,2);
% we used this values of W = [w0 w1] true values to create the data
w0 = 1; % Parameters of the actual underlying model that we wish to recover
w1 = 1;  % We will estimate these values with w0 and w1 respectively.


%% Gaussian additive noise
noiseSD = 0.25;          % Standard deviation of Gaussian noise, applied to actual underlying model.
likelihoodPrecision = 1/(noiseSD)^2;

%% gaussian Prior

priorMu = [0;0];
S = 2.0;   
priorSigma = S^2.*eye(2);  %Covariance Matrix
%%for MiniProjecttask1 partg 


%%
%% Calculate Wmle
Wmle = inv((Xtrain)'*(Xtrain))*(Xtrain'*Ytrain);
Wmap = zeros(2,m);
Cwt = zeros(2,2,m);
%% Plot the prior distribution over w0, w1
%figure(1);
priorPDF = @(W)gausspdf(W,priorMu',priorSigma);
%contourPlot(priorPDF,[],Wmle);

%%
% figure(2);
[postW,postMu,postSigma] = NextPDF([1,Xtrain(1)],Ytrain(1),likelihoodPrecision,priorMu,priorSigma);
%    contourPlot(postW,[1,1],Wmle,postMu);
Xtrainnew = Xtrain(1,:);
 Ytrainnew = Ytrain(1,:);

%% store all values of Wmap and Sigma for task 1 part g
Wmap(:,1) = postMu;
 
Cwt = postSigma;
%%
for i = 1 : 999
   %h(i)= figure(i+1);
  
  
   [postW,postMunew,postSigmanew] = NextPDF(Xtrainnew,Ytrainnew,likelihoodPrecision,postMu,postSigma);
  % contourPlot(postW,[1,1],Wmle); 
  postSigma= postSigmanew;
  postMu=postMunew;
   Xtrainnew = [Xtrainnew;Xtrain(i+1,:)];
   Ytrainnew = [Ytrainnew;Ytrain(i+1)];
    %% save the files
  
   filename = sprintf('test_image_%d.png', i);
  %saveas(h(i),filename);
  
   
   
  %% store all values of Wmap and Sigma for task 1 part g
  
  Wmap(:,i+1) = postMunew;
  Cwt(:,:,i+1) = postSigmanew;
  
  %%
 
end
%%


%% prediction Interval

Sigmapartb = zeros(m,1);
for i = 1: m
  Sigmapartb(i,1) = sqrt((1/4).^2+(Xtrain(i,:)*(Cwt(:,:,i))*Xtrain(i,:)'));

end


Tmin = zeros(1,m);
Tmax = zeros(1,m);
% Xtrainnewpred = Xtrain(1,:);
% Tmin(1,1) = (Wmap(:,1)' * Xtrainnewpred(1,:)') - (Sigmapartb(1,1)*1.64);
% Tmax(1,1) = (Wmap(:,1)' * Xtrainnewpred(1,:)') + (Sigmapartb(1,1)*1.64);

% for i = 1: m-1
%     
%    Tmin(1,i+1) = Tmin(1,i) + Wmap(:,1)' * Xtrainnew(i+1,:)'; 
%    Tmax(1,i+1) = Tmax(1,i) + Wmap(:,1)' * Xtrainnew(i+1,:)';  
% end

for i = 1: m-1
   
    Tmin(1,i) = (Wmap(:,i)' * Xtrainnewpred(i,:)') - (Sigmapartb(i,1)*1.64);
    Tmax(1,i) = (Wmap(:,i)' * Xtrainnewpred(i,:)') + (Sigmapartb(i,1)*1.64);
      Xtrainnewpred = [Xtrainnewpred;Xtrain(i+1,:)];

 end

for i=1:m
plot(X(i,1),X(i,2),'y*',X(i,1),Tmin(1,i),'b.',X(i,1),Tmax(1,i),'b.');
legend('data','tmin','tmax');
hold on
drawnow
end

