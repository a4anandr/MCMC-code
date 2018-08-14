%% Code for variance paper - Metropolis Hasting algorithm

clear all;
close all;

display_figure = 0;      % 0 for No display, 1 for display

% s=rng('default');
tic;

%% Declaring variables
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
Y=X;

% For histogram
global N;     % For Monte Carlo simulations
N =10000;

% Subexponential density
delta = 1;    % 0 < delta < 1
% density  =  (abs((x - mu)/sig))^(2* (delta - 1)) * exp(- 0.5 * (abs(x-mu)/sig)^(2*delta));  % Weibull term 
density  = exp(- 0.5 * (abs(x-mu)/sig)^(2*delta));
density_x = matlabFunction(density);

%% sigma_B and U for Langevin diffusion
global sigmaB;
sigmaB = 0.4;
mut1 = -1;
sigt1 = sigmaB;
wt1 = 0.5;
mut2 = 1;
sigt2 = sigmaB;
wt2 = 1-wt1;

% p(x)= w1*p1(x)+w2*p2(x)
p1   = density_x(mut1,sigt1,x);
p2   = density_x(mut2,sigt2,x);
p    = wt1*p1 + wt2*p2;
p_mf = matlabFunction(p);

% Defining potential function U(x)
Utot  = -log(p);
Utot_x  = matlabFunction(Utot);
del_Utot = diff(Utot);
del_Utot_x= matlabFunction(del_Utot);

% Observation process
% c = sin(x);
c = x;
c_x = matlabFunction(c);
% grad_c = 1;
grad_c = diff(c);
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

% figure(1)
% plot(X,K_fpf,'r','linewidth',2.0);
% hold on;

%% Basis Functions
% Defining the new set of basis functions x^k * p_m(x) (Polynomials around each density)
% for d = 1:1:20;
variance_reduction(1) = 1;

% for d = 10:10:20
    % d
    d = 20;            % Imposing restriction that d can only be even, d0 = d/2 is the number of basis functions around each p_i
    d0 = d/2;    
    p1_basis = exp(-(x-mut1)^2/(2   * 2 *  sigt1^2));    % factor a 
    p2_basis = exp(-(x-mut2)^2/(2   * 2 *  sigt2^2));  
    % p1_basis = density_x(mut1,sqrt(1)*sigt1,x);
    % p2_basis = density_x(mut2,sqrt(1)*sigt2,x);
    
    % p1_basis  = p1;
    % p2_basis  = p2;    
    for k=0:1:d0-1
        psi(k+1)    = x^k * p1_basis;
    end
    for k= d0+1:1:d
        psi(k)      = x^(k-(d0+1)) * p2_basis;
    end
    
    if display_figure == 1
        psi_d = reshape(psi_x(X),length(psi_x(X))/d,d);
        for i = 1:1:d
            plot(X,psi_d(:,i),'r');
            hold on;
        end
    end
    
    grad_psi = diff(psi);
    psi_ddot = diff(grad_psi);
    Dpsi     = - del_Utot * grad_psi + psi_ddot;

    psi_x = matlabFunction(psi); 
    grad_psi_x = matlabFunction(grad_psi);
    Dpsi_x = matlabFunction(Dpsi);
    
%% Method 1 using Langevin generator - To minimize the asymptotic variance
   % M_T
   M_t_mf = matlabFunction(grad_psi' * grad_psi *p);
   M_t_lang = 0;
   for i = 1:1:length(X)
       M_t_lang = M_t_lang + M_t_mf(X(i));
   end
   
   % b_T 
   c_tilde = c - eta; 
   b_t_mf = matlabFunction(c_tilde * psi * p);
   b_t_lang  = sum(reshape(b_t_mf(X),length(X),d))';
   
   theta_lang = M_t_lang \ b_t_lang;
    
% Computation of new estimator and control variate for method 1
   [ c_theta_lang_x, call_no] = cv_from_theta(c,theta_lang, del_Utot, grad_psi);    % New estimator function
   
%% Method 2 using Langevin generator - To minimize the ordinary variance
%    M_t_mf2 = matlabFunction(Dpsi' * Dpsi *p);
%    M_t_lang_var = 0;
%    for i = 1:1:length(X)
%        M_t_lang_var = M_t_lang_var + M_t_mf2(X(i));
%    end
%    
%    % b_T
%    b_t_mf2 = matlabFunction(- c_tilde * Dpsi * p);
%    b_t_lang_var  = sum(reshape(b_t_mf2(X),length(X),d))';
%    
%    theta_lang_var = M_t_lang_var \ b_t_lang_var;
%     
% % Computation of new estimator and control variate for method 2
%    c_theta_lang2_x = cv_from_theta(c,theta_lang_var, del_Utot, grad_psi);    % New estimator function
%    
%% Method 3 using an approximation to the new method using Langevin generator for small values of gamma - To minimize the asymptotic variance
%    gamma   = [0.005 0.01 0.05 0.1 0.5 1];
gamma = 0.1;
%    M_t_rwm_approx = -(gamma)^2 * M_t_lang_var + 2 * gamma * M_t_lang;
%    b_t_rwm_approx =  gamma * ( -b_t_lang_var + 2 * b_t_lang) ;                     % Actually needs to have a - sign
%    
%    theta_rwm_approx = M_t_rwm_approx \ b_t_rwm_approx;
%     
% % Computation of new estimator and control variate for method 3
%    c_theta_rwm_approx_x = cv_from_theta(c,theta_rwm_approx, del_Utot, grad_psi);    % New estimator function


%% Metropolis Hastings algorithm for sampling
% RWM with step-size parameter \gamma 
% rng(N);
% Time domain specifications
T = 100000;
burnin = 2;

Phi = randn;                   %  Initial sample for Phi(0)
for i = 1:1:N
    i  
    count = 0;
    for k = 1:1:length(gamma)
%        gamma(k)

%         Mhat_t_rwm = zeros(d,d);
%         b_t_rwm = zeros(d,1);
        
        M_t_lang2 = zeros(d,d);
        b_t_lang2 = zeros(d,1);
        
%         Mhat_t_rwm2 = zeros(d,d);
%         b_t_rwm2 = zeros(d,1);
        
        t = 1;   
        while t < T
            Phi_next = Phi(t) + sqrt(2 * gamma(k)) * randn;         % Possible next sample obtained using RW from the current sample Phi(t)
   % Acceptance ratio  
            alpha = p_mf(Phi_next)/p_mf(Phi(t));
            t = t+1;
            if(rand < alpha)
                Phi(t) = Phi_next;
            else
                Phi(t) = Phi(t-1);
            end
            % Monte Carlo estimate for M and b
            if t > burnin    % t > 2
                count = count + 1;                
%                 Mhat_t_rwm = Mhat_t_rwm + (1/(T-burnin)) * (gamma(k)^(-1) * psi_x(Phi(t-2))' - gamma(k)^(-1) * psi_x(Phi(t))') * gamma(k)^(-1) * psi_x(Phi(t-2));
%                 b_t_rwm    = b_t_rwm + (1/(T + 1 - burnin)) * (c_x(Phi(t-1)) - eta) * (gamma(k)^(-1) * psi_x(Phi(t-1))' + gamma(k)^(-1) * psi_x(Phi(t))');
                
                M_t_lang2 = M_t_lang2 + (1/(T - burnin)) * (grad_psi_x(Phi(t))'* grad_psi_x(Phi(t))); 
                b_t_lang2 = b_t_lang2 + (1/(T - burnin)) * (c_x(Phi(t)) - eta) * psi_x(Phi(t))';
                
              
%                 Mhat_t_rwm2 = Mhat_t_rwm2 + (1/(T - burnin)) * (2 * gamma(k) *(Dpsi_x(Phi(t))'* psi_x(Phi(t))) + gamma(k)^2  * Dpsi_x(Phi(t))'* Dpsi_x(Phi(t))); 
%                 b_t_rwm2    = b_t_rwm2 - (1/(T - burnin)) * (2 * gamma(k) * (c_x(Phi(t)) - eta) * psi_x(Phi(t))' + gamma(k)^2 * (c_x(Phi(t)) - eta) * Dpsi_x(Phi(t))');
                
            end
        end             
%         M_t_rwm = (1/2) * (Mhat_t_rwm + Mhat_t_rwm');
%         theta_rwm =  M_t_rwm \ b_t_rwm;
%         [c_theta_rwm_x,call_no] = cv_from_theta(c,theta_rwm, del_Utot, grad_psi, call_no);    % New estimator function 
        
        theta_lang2 = M_t_lang2 \b_t_lang2;
        [c_theta_lang2_x,call_no] = cv_from_theta(c,theta_lang2, del_Utot, grad_psi, call_no-1);
        
%         M_t_rwm2 = (1/2) * (Mhat_t_rwm2 + Mhat_t_rwm2');
%         theta_rwm2 = M_t_rwm2 \ b_t_rwm2;
%         [c_theta_rwm2_x,call_no] = cv_from_theta(c,theta_rwm2, del_Utot, grad_psi, call_no-1);    % New estimator function 
        
%        ratio_theta(k) = theta_rwm2(1)/theta_rwm(1);
        
%         figure(call_no)
%         plot(X,K_fpf,'r','linewidth',2.0);
%         
%         figure(100+call_no)
%         plot(X,p_mf(X),'r');
%         hold on;
%         histogram(Phi(burnin : end),'Normalization','pdf');
             
    end
    eta_nocv(i)    = mean(c_x(Phi(burnin+1 : end)));
    eta_lang(i)    = mean(c_theta_lang_x(Phi(burnin+1 : end)));  
%    eta_rwm(i)     = mean(c_theta_rwm_x(Phi(burnin+1 : end)));
    eta_lang2(i)   = mean(c_theta_lang2_x(Phi(burnin+1 : end)));
%    eta_rwm2(i)     = mean(c_theta_rwm2_x(Phi(burnin+1 : end)));
end
% end

%% Histogram and asymptotic variance computation
eta_hat      = mean(eta_nocv)
eta_hat_lang = mean(eta_lang)
% eta_hat_rwm  = mean(eta_rwm)
eta_hat_lang2= mean(eta_lang2)
% eta_hat_rwm2  = mean(eta_rwm2)

hist_variance_lang(d,:)   = sqrt((T - burnin))*(eta_lang - eta);
% hist_variance_rwm(d,:)    = sqrt((T - burnin))*(eta_rwm - eta);
hist_variance_nocv(d,:)   = sqrt((T - burnin))*(eta_nocv - eta);
hist_variance_lang2(d,:)  = sqrt((T - burnin))*(eta_lang2 - eta);
% hist_variance_rwm2(d,:)   = sqrt((T - burnin))*(eta_rwm2 - eta);

[muhat_lang, sigmahat_lang] = normfit(hist_variance_lang(d,:));
%[muhat_rwm,  sigmahat_rwm]  = normfit(hist_variance_rwm(d,:));
[muhat_nocv, sigmahat_nocv] = normfit(hist_variance_nocv(d,:));
[muhat_lang2,sigmahat_lang2]= normfit(hist_variance_lang2(d,:));
% [muhat_rwm2,  sigmahat_rwm2]  = normfit(hist_variance_rwm2(d,:));

sigmahat_nocv
sigmahat_lang
% sigmahat_rwm
sigmahat_lang2 
% sigmahat_rwm2

%% Figures
% Histogram of the RWM
if display_figure == 1
scale = (1/100);
[dist,loc]= hist(Phi, scale * T);
figure;   
% histogram(Phi, scale * T, 'Normalization', 'pdf');
hist(Phi, scale * T);
hold on;
plot(X,p_mf(X)*(max(max(dist))/max(p_mf(X))),'r','linewidth',2.0);
% plot(X,p_mf(X),'r','linewidth',2.0);
title('Histogram of the particles with density p');

% Histograms of estimates with computed asymptotic variance 
figure;
span = ceil(max(hist_variance_lang(d,:)))-floor(min(hist_variance_lang(d,:)));
spacing = 1;
bins_1 = span/spacing;
[weight, loc] = hist(hist_variance_lang(d,:),bins_1);
histfit(hist_variance_lang(d,:),bins_1);
hold on;
title(['Histogram of \eta^{i} - \eta for T =' num2str(T)]);
legend('Histogram','Gaussian fit','Actual Gaussian');

figure;
span = ceil(max(hist_variance_rwm(d,:)))-floor(min(hist_variance_rwm(d,:)));
spacing = 1;
bins_2 = span/spacing;
[weight, loc] = hist(hist_variance_rwm(d,:),bins_2);
histfit(hist_variance_rwm(d,:),bins_2);
hold on;
title(['Histogram of \eta^{i} - \eta for T =' num2str(T)]);
legend('Histogram','Gaussian fit','Actual Gaussian');

figure;
span_0 = ceil(max(hist_variance_nocv(d,:)))-floor(min(hist_variance_nocv(d,:)));
spacing_0 = 1;
bins_0 = span_0/spacing_0;
[weight, loc_0] = hist(hist_variance_nocv(d,:),bins_0);
histfit(hist_variance_nocv(d,:),bins_0);
hold on;
title(['Histogram of \eta^{i} - \eta for T =' num2str(T)]);
legend('Histogram','Gaussian fit','Actual Gaussian');
% save(['hist_d_0' num2str(d) 'N_' num2str(N) 'T_' num2str(T) '.mat'],'hist_variance_0','sigmahat');

figure;
hist(hist_variance_nocv(d,:),bins_0);
h = findobj(gca,'Type','patch');
h.FaceColor = 'r';
hold on;
hist(hist_variance_lang(d,:),bins_1);
% hist(hist_variance_rwm(d,:),bins_2);
legend('No CV',['d=' num2str(d)]);
title('Histogram for asymptotic variance comparison');
end

% variance_reduction(d/2 + 1) = sigmahat^2 / sigmahat_0^2 
% end

% figure;
% plot(0:2:20,variance_reduction,'b^','linewidth',2);
% title(['Relative asym variance reduction - Metropolis Hastings for delta', num2str(delta)]);
% xlabel('d ->');
% ylabel('\sigma^2/\sigma_0^2');

toc

