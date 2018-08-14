function [ c_tilde_theta_x, call_no ] = cv_from_theta(c,theta_opt, del_Utot, grad_psi, call_no);
% Function that computes the new estimator with control variates -
% c_tilde_theta_x using the optimal parameters theta* for a given basis psi
% and its gradient grad_psi
if nargin == 4
   call_no  = 1;
else 
   call_no  = call_no + 1;
end

d = length(grad_psi);
K_theta = 0;
for i = 1:1:d
    K_theta = K_theta + theta_opt(i) * grad_psi(i);
end

K_dot_theta = diff(K_theta);
cv_theta  = - del_Utot * K_theta +  K_dot_theta;    % This should be close to c(x) + control variate
c_tilde_theta  = c + cv_theta;                      % Difference between c and c_theta, new estimator 
c_tilde_theta_x  = matlabFunction(c_tilde_theta);    

% To plot the gain approximation function
K_theta_x = matlabFunction(K_theta);
X = -3:0.05:3;
% figure(call_no)
% hold on;
% plot(X,K_theta_x(X));
% xlim([-2 2]);

end

