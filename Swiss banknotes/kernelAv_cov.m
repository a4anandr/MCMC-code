function [thetaav, b ] = kernelAv_cov(theta,z)
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
rng(100);
[n, d]  = size(theta);
train_i = randsample(n, 200);    % 0.0025*n
train_n = length(train_i);
theta_train = theta(train_i,:);
theta_mean = mean(theta,1); 
M = zeros(train_n,train_n);
b = zeros(train_n,d);


% Gaussian kernel and its derivatives
lambda  = 1e-7;   % 1e-7, 0 or small value is optimal
epsilon = 2;  % So far 1.5 - 2 is the best
% Kernel definitions using symbolic variables
% ker     = @(x,y) exp(-norm(x - y)^2/(4 * epsilon));
% ker_x   = @(x,y) -(x - y)/ (2 * epsilon) * exp(-norm(x - y)^2/(4 * epsilon));
% lap_ker = @(x,y) -(1/(2 * epsilon)) * ( d - norm(x - y)^2/(2 * epsilon)) * exp(-norm(x - y)^2/(4 * epsilon));

% Mnew = 0;
for i = 1 : n 
%    i
    for j =  1 : train_n
        K(i,j) = exp(-(norm(theta(i,:) - theta_train(j,:))^2/(4 * epsilon)));
        for k = 1 : d
            K_xnew(j,k,i) = -((theta(i,k) - theta_train(j,k))/ (2 * epsilon)) * K(i,j);          % 100 * 4 * 10000
        end
        lap(i,j) = -(1/(2 * epsilon)) * (d - (norm(theta(i,:) - theta_train(j,:))^2/(2 * epsilon))) * K(i,j);           % need to verify this
        DK(i,j)  = - 2 * z(i,:) * K_xnew(j,:,i)' + lap(i,j);
    end
end

covb = cov([K theta]);
b    = covb(1:train_n,train_n+1:train_n+d);
covM = cov([K -DK]);
M    = lambda * K(train_i,:) + covM(1:train_n,train_n+1:2*train_n);
beta = M \ b;

for i = 1: 1: n  
    thetaav(i,:) = theta(i,:) + DK(i,:) * beta;
end
toc;
end

