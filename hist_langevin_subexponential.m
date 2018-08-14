%% Code for variance paper - Histogram of the variance for a fixed value of basis order d 
clc
clear all;
close all;

basis = 1;              % 0 for old, 1 for new (even)
display_figure = 0;      % 0 for No display, 1 for display
tic;

% Declaration of global variables
global sigmaB;
global N; % For Monte Carlo simulations

% Process and observation variance
sigmaB=0.4;

% Declaration of symbolic variables
syms x;
syms y;
syms sig;
syms mu;
syms eps;

% State space domain
xmin = -3;
xmax = 3;
step = 0.005;
X=xmin:step:xmax;


% Time domain specifications
dt = 0.1;
sdt = sqrt(dt);
T = 10000;
sqrt2 = sqrt(2);

% For histogram
N = 10000;

% Subexponential density
delta = 1;    % 0 < delta < 1
% density  =  (abs((x - mu)/sig))^(2* (delta - 1)) * exp(- 0.5 * (abs(x-mu)/sig)^(2*delta));  % Weibull term 
density  = exp(- 0.5 * (abs(x-mu)/sig)^(2*delta));   % unnormalized density 
density_x = matlabFunction(density);

%% U and sigmaB for Langevin diffusion
mut1= -1;
sigt1=sigmaB;
wt1=0.5;
mut2=1;
sigt2=sigmaB;
wt2=1-wt1;

% p(x)= w1*p1(x)+w2*p2(x)
p1   = density_x(mut1,sigt1,x);
p2   = density_x(mut2,sigt2,x);
p    = wt1*p1 + wt2*p2;
p_mf   = matlabFunction(p);

% Defining potential function U(x)
Utot_x  = -log(p);
Utot_mf  = matlabFunction(Utot_x);
del_Utot = diff(Utot_x);
del_Utot_mf= matlabFunction(del_Utot);

% Observation function
c = x;
% c = sin(x);
c_x = matlabFunction(c);
grad_c = 1;
% grad_c = diff(c);
cp = matlabFunction(c*p);
eta = sum(cp(X) * step);

%% Computation of the exact solution to Poisson's equation - K_act, This will be used to come up with the theoretical value of asymptotic variance, 2 <K, K>.
i=1;
for xi = xmin : step : xmax
    integral(i) = 0;
    for xj = xi : step : xmax + 10
        integral(i) = integral(i) + p_mf(xj) * (c_x(xj) - eta) * step;
    end
    K_fpf(i) = integral(i) / p_mf(xi);
    i = i + 1;
end
gamma2_0 = 2 * sum(K_fpf .* K_fpf .* p_mf(X) * step);


%% Basis Functions
variance_reduction = 1;
theoretical_var_reduction = 1;
gamma2_d = gamma2_0;
l = [6 10 20]; 

for d = 1 : 1 : length(l) 
    % d = 20;
    l(d)
    d0 = l(d)/2;

    p1_basis = exp(-(x-mut1)^2/(2  * 2 * sigt1^2));
    p2_basis = exp(-(x-mut2)^2/(2  * 2 * sigt2^2));
    % p1_basis = density_x(mut1, sqrt(1) * sigt1,x);
    % p2_basis = density_x(mut2, sqrt(1) * sigt2,x);
    % p1_basis = p1;
    % p2_basis = p2;

if basis == 0
    for k=0:1:uint8(l(d)/2)-1
        psi(k+1)      = x^k * p1_basis;
    end
    for k=uint8(l(d)/2):1:l(d)-1
        psi(k+1)      = x^(k-uint8(l(d)/2)) * p2_basis;
    end
        grad_psi = diff(psi);
        
   if Kalman_offset == 1  
      d = d + 1;
      psi(k+1) = x;                              % Extra basis function for the constant offset
      grad_psi(k+1)=(x+1)-(1+1e-10)*x;           % Constant offset required for Kalman gain
   elseif Kalman_offset == 2
      d = d + 1;
      psi(k+1) = K_kal * x;                            
      grad_psi(k+1)=(x+ K_kal)-(1 + 1e-10)*x;
   end
   
   psi_x = matlabFunction(psi); 
   grad_psi_x = matlabFunction(grad_psi);

else       
    d0 = l(d)/2;
    for k=0:1:d0-1
        psi(k+1)      = x^k * p1_basis;
    end
    for k= d0+1:1:l(d)
        psi(k)      = x^(k-(d0+1))*p2_basis;
    end
    grad_psi = diff(psi);
    psi_ddot = diff(grad_psi);
    
    psi_x = matlabFunction(psi); 
    grad_psi_x = matlabFunction(grad_psi);
    
    if display_figure == 1
       psi_d = reshape(psi_x(X),length(psi_x(X))/l(d),l(d));
       for i = 1:1:l(d)
          plot(X,psi_d(:,i),'r');
          hold on;
       end
    end
end

%% Optimal weights theta* - By integration
% M_T
M_t = grad_psi' * grad_psi *p;
M_t_mf = matlabFunction(M_t);
M_t_v = 0;
for i = 1:1:length(X)
    M_t_v = M_t_v + M_t_mf(X(i));
end

%b b_T
c_tilde = c -  eta;        
b_t = c_tilde * psi * p;
b_t_mf = matlabFunction(b_t);
b_t_v = sum(reshape(b_t_mf(X),length(X),l(d)))';
  
theta_opt = M_t_v\b_t_v;
    
%% Computation of control variates and new estimator
K_theta = 0;
for i = 1:1:l(d)
   K_theta = K_theta + theta_opt(i) * grad_psi(i);
end 

K_dot_theta   = diff(K_theta);
cv_theta      = del_Utot * K_theta -  K_dot_theta;     % This should be close to c(x)
c_tilde_theta = c - cv_theta;                          % Difference between c and c_theta - Control variate cv
   
K_theta_x        = matlabFunction(K_theta);
c_tilde_theta_x  = matlabFunction(c_tilde_theta);

% Theoretical variance with control variates
% gamma2_d (d/2 + 1) = 2 * sum((K_fpf - K_theta_x(X)).*(K_fpf-K_theta_x(X)).*p_mf(X) * step);
    
%% Langevin diffusion for sampling
Phi = randn;
eta_cv(1) = c_tilde_theta_x(Phi);
eta_c(1)  = c_x(Phi);
for i = 1:1:N
    i
    rng(i + N)
    for t=1:1:T/dt
        Phi(t+1) = Phi(t) - del_Utot_mf(Phi(t))*dt +  sqrt2 * sdt * randn;
        % eta_cv(t+1) = (t/(t+1)) * eta_cv(t) + (1/(t+1)) * c_tilde_theta_x(Phi(t+1));  % Not sure if dt is required
        % eta_c(t+1)  = (t/(t+1)) * eta_c(t)  + (1/(t+1)) * c_x(Phi(t+1));  
    end
    eta_cv(i)   = mean(c_tilde_theta_x(Phi));
    eta_nocv(i) = mean(c_x(Phi));
    % eta_i(i) = eta_cv(end);
    % eta_0_mean(i) = eta_c(end);
end

%% Histogram and asymptotic variance computation
hist_variance(d,:)   = sqrt(T)*(eta_cv - eta);
hist_variance_0(d,:) = sqrt(T)*(eta_nocv - eta);
[muhat, sigmahat]  = normfit(hist_variance(d,:));
[muhat_0, sigmahat_0] = normfit(hist_variance_0(d,:));

%% Figures
if display_figure == 1
% Histogram of the Langevin diffusion particles
[dist,loc]= hist(Phi, 0.01*(T/dt));
figure;   
hist(Phi, 0.01*(T/dt));
hold on;
plot(X,p_mf(X)*(max(max(dist))/max(p_mf(X))),'r','linewidth',2.0);
title('Histogram of the particles with density p');

% Histogram of eta_i - eta_hat
figure;
span = ceil(max(hist_variance(d,:)))-floor(min(hist_variance(d,:)));
spacing = 0.1;
bins = span/spacing;
[weight, loc] = hist(hist_variance(d,:),bins);
histfit(hist_variance(d,:),bins);
hold on;
title(['Histogram of \eta^{i} - \eta for T =' num2str(T)]);
legend('Histogram','Gaussian fit');
% save(['hist_d_' num2str(d) 'N_' num2str(N) 'T_' num2str(T) '.mat'],'hist_variance','sigmahat');

% Histogram for d = 0
figure;
span_0 = ceil(max(hist_variance_0(d,:)))-floor(min(hist_variance_0(d,:)));
spacing_0 = 0.1;
bins_0 = span_0/spacing_0;
[weight, loc_0] = hist(hist_variance_0(d,:),bins_0);
histfit(hist_variance_0(d,:),bins_0);
hold on;
title(['Histogram of \eta^{i} - \eta for T =' num2str(T)]);
legend('Histogram','Gaussian fit');
% save(['hist_d_0_' num2str(d) 'N_' num2str(N) 'T_' num2str(T) '.mat'],'hist_variance_0','sigmahat_0');

figure;
hist(hist_variance_0(d,:),bins_0);
h = findobj(gca,'Type','patch');
h.FaceColor = 'r';
hold on;
hist(hist_variance(d,:),bins);
legend('No CV',['d=' num2str(d)]);
title('Langevin');
end

log(abs(eig(M_t_v)));
% variance_reduction(d/2 + 1) = sigmahat^2 / sigmahat_0^2 
% theoretical_var_reduction(d/2 + 1) = gamma2_d (d/2 + 1)/gamma2_0
toc

end

%figure;
% plot(0:2:20,variance_reduction,'r*','linewidth',6);
% hold on;
% plot(0:2:20,theoretical_var_reduction,'bo','linewidth',6);
% title(['Relative asym variance reduction - Langevin for \delta =' num2str(delta)]);
% legend('Simulation','Theoretical');
xlabel('d ->');
ylabel('\sigma^2/\sigma_0^2');




