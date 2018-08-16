function [thetaZv, a] = quadraticAv_cov(theta, z)
% quadraticAv_cov: calculation of diff TD control variates using covariance
% function to make it computationally faster
% Written by Anand Radhakrishnan
% (c) UF, Department of ECE., All rights reserved.
% 2z = grad_U (- grad log posterior) 
tic;
[n, d] = size(theta);

zSquare = zeros(n, d*(d+3)/2);
k = 2*d+1;

psi = [theta 0.5*theta.^2 ];
zSquare(:, 1:d) = z;
zSquare(:, (d+1):(2*d)) = z.*theta-1/2;
for i = 1:(d-1)
  for j = (i+1):d
    zSquare(:, k) = theta(:, i).*z(:, j) + theta(:, j).*z(:, i);
    psi(:,k) = theta(:, i) .* theta(:, j);
    k = k+1;
  end
end

[b, a]   = deal(zeros(d*(d+3)/2, d));

covAll   = cov([2*zSquare psi theta]);
M        = covAll(1:d*(d+3)/2, d*(d+3)/2 + 1 : d*(d+3)); 

b      = covAll(d*(d+3)/2 + 1 : d*(d+3),d*(d+3)+1: d*(d+4));
a = M\b;

thetaZv = theta - 2 * zSquare * a;
toc;
