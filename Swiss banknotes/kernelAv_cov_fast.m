function [thetaav, b ] = kernelAv_cov_fast(theta,z)
% Written by Anand Radhakrishnan
% Application of RKHS methods for the available datasets. 
% Given a large number of samples, fit a Gaussian kernel function to a
% subset of these samples and solve a regularized empirical risk
% minimization problem and obtain the optimal parameter weights for the
% control variates 
% 1. Construct kernel functions on the subset of samples, this is in 4
% dimensions, i.e. exp(-|| theta_1 - theta_2||^2/ 4*epsilon)
% 2. 

tic; 
[n, d]  = size(theta);
train_i = randsample(n, 100);    % 0.0025*n
train_n = length(train_i);
theta_train = theta(train_i,:);
theta_mean = mean(theta,1); 
M = zeros(train_n,train_n);
b = zeros(train_n,d);


% Gaussian kernel and its derivatives
lambda  = 1e-7;   % 1e-6 or small value is optimal
epsilon = 2;  % So far 1.5 - 2 is the best

tic
for i = 1 : n 
    for j =  1 : train_n
        norm_matrix(i,j) = norm(theta(i,:) - theta_train(j,:))^2;
    end
end
K  = exp(-norm_matrix / (4 * epsilon));
for j = 1 : train_n
    for k = 1 : d
        K_xnew(:,j,k) = -((theta(:,k) - theta_train(j,k) * ones(n,1))/ (2 * epsilon)) .* K(:,j);          % 100 * 4 * 10000
    end
end
lap = -(1/(2 * epsilon)) * (d * ones(n,train_n) - ( norm_matrix / (2 * epsilon))) .* K;
DK  = - 2 * permute(sum( z .* permute(K_xnew,[1 3 2]),2),[1 3 2]) + lap;

covb = cov([K theta]);
b    = covb(1:train_n,train_n+1:train_n+d);
covM = cov([K -DK]);
M    = lambda * K(train_i,:) + covM(1:train_n,train_n+1:2*train_n);
beta = M \ b;

thetaav = theta + DK * beta;

toc
end

