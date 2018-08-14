function [thetaZv, a] = quadraticAv_cov(theta, z)
% quadraticAZv: calculation of ZV-RMHMC estimates using a quadratic
% polynomial.

% Written by Anand Radhakrishnan
% (c) UF, Department of ECE., All rights reserved.
tic;
[n, d] = size(theta);
theta_mean = mean(theta); 

Mold = zeros(d*(d+3)/2,d*(d+3)/2);
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

% for m = 1:1:n    
%    grad_psi(:,:,m) = [eye(d); diag(theta(m,:)); theta(m,2) theta(m,1) 0 0; theta(m,3) 0 theta(m,1) 0; theta(m,4) 0 0 theta(m,1);0 theta(m,3) theta(m,2) 0; 0 theta(m,4) 0 theta(m,2); 0 0 theta(m,4) theta(m,3)];   % 14 * 4
%    % Dpsi(m,:) = - 2 * z(m,:) * [eye(d) diag(theta(m,:)) [ theta(m,2) theta(m,3)  theta(m,4) 0 0 0; theta(m,1) 0 0 theta(m,3) theta(m,4) 0;  0 theta(m,1) 0 theta(m,2) 0 theta(m,4); 0 0 theta(m,1) 0  theta(m,2) theta(m,3)]] + [ 0 0 0 0 1 1 1 1 0 0 0 0 0 0]; 
%    Mold = Mold + (1/(n-1)) * grad_psi(:,:,m) * grad_psi(:,:,m)';
% end
% 
% cond(Mold)
% for i = 1:d
%    bold(:, i)  = (1/n)* psi' *(theta(:,i) - theta_mean(i));   
% end
% aold = Mold \ bold;


covAll   = cov([2*zSquare psi theta]);
M      = covAll(1:d*(d+3)/2, d*(d+3)/2 + 1 : d*(d+3)); 
cond(M)
b      = covAll(d*(d+3)/2 + 1 : d*(d+3),d*(d+3)+1: d*(d+4));
a = M\b;

thetaZv = theta - 2 * zSquare * a;
toc;
