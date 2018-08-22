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
train_n = 200;
check   = 40;           % Initializing at 10^40, assigned after observing from a number of trials, the condition number is usually ~ 10^20 for good results
iteration = 1;          % To count the number of iterations it takes to get training samples that give a lower condition number.
theta_mean = mean(theta,1); 
% theta_mean = [-0.7117    0.7969    0.9974    3.0062];   % Just hard coding the mean
M = zeros(train_n,train_n);
b = zeros(train_n,d);


% Gaussian kernel and its derivatives
lambda  = 1e-7;   % 1e-6 or small value is optimal
epsilon = 2;  % So far 1.5 - 2 is the best
tic
v = version('-release');
while(check > 21)
    train_i     = randsample(n, train_n);    
    theta_train = theta(train_i,:);
    if v == '2018a'
        for j  = 1 : train_n
            norm_matrix(:,j) = vecnorm((theta - repmat(theta_train(j,:),n,1)),2,2).^2;
        end
    else
        for i = 1 : n
            for j =  1 : train_n
                norm_matrix(i,j) = norm(theta(i,:) - theta_train(j,:))^2;
            end         
        end
    end
    check = log10(cond(norm_matrix(train_i,:)));           % To ensure that highly similar samples are not chosen in the training set. This will cause high condition number of the matrix and in turn give poor results.
    iteration = iteration + 1;
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

