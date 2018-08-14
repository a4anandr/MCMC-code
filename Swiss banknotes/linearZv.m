function [thetaZv, a] = linearZv(theta, z)
% linearZv: calculation of ZV-RMHMC estimates using a linear polynomial.

% Written by Theodore Papamarkou.
% (c) UCL, Department of Statistical Science. All rights reserved.
tic;

[~, d] = size(theta);

covAll = zeros(d+1, d+1, d);
Sigma = zeros(d, d, d);
[sigma, a] = deal(zeros(d));

for i = 1:d
   covAll(:, :, i) = cov([z theta(:, i)]);
   Sigma(:, :, i) = inv(covAll(1:d, 1:d, i));
   sigma(:, i) = covAll(1:d, d+1, i);
   a(:, i) = -Sigma(:, :, i)*sigma(:, i);
end

thetaZv = theta+z*a;
toc;
