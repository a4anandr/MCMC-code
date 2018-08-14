function [thetaav, a ] = linearAv(theta,z)
% Written by Anand Radhakrishnan
% Function that takes in the MCMC samples and its log-gradient and then computes parameter values a and the new estimate  
%   Detailed explanation goes here
% 2*z = \nabla U - To be used for computing the control variates
% Linear basis in R^4 is chosen; i.e. h_a  = a^\transpose \psi, where
% \psi(i) = [z1(i) z2(i) z3(i) z4(i)] for i = 1 to 10000
% Is M = 1 in this case? 

tic; 
[n, d]     = size(theta);
theta_mean = mean(theta,1); 
a = zeros(d);

% Non vectorized form
% for k = 1:d
%    for m  = 1:1:N
%        a(:,k)  = a(:,k) + (1/N) * ((theta(m,k) - theta_mean(k))* (theta(m,:)'));   
%    end
% end

% for m = 1 : 1: N
%     for k = 1:d
%         thetaav(m,k) = theta(m,k) - 2 * z(m,:) * a(:,k); 
%     end
% end

% Vectorized form
% for m  = 1:1:n
%     a  = a + (1/n) * (repmat((theta(m,:) - theta_mean),d,1) .* repmat(theta(m,:)',1,d));   
% end

for i = 1:d
   a(:, i)  = (1/n)* theta' *(theta(:,i) - theta_mean(i));   
end

thetaav = theta - 2 * z * a; 

toc;



