%% Main function to demonstrate asymptotic variance reduction in Markov chain Monte Carlo algorithms 
% - Obtains samples from an MCMC algorithm - i) Langevin ii)
% Metropolis-Hastings
% - Passes these particles Xi to the different gain function approximation
% functions -  i) finite dim, ii) Coifman
% kernel, iii) RKHS based algorithm iv) constant gain approximation
% - Obtains approximations to the value function (solution to Poisson's
% equation) and computes the control variates corresponding to each
% particle/sample Xi
% - Run independent trials and plot the histograms to show the variance
% reduction achieved. 

clear;
clc;
close all;


syms x;
diag_main = 1;   % Diagnostics flag for main function, displays figures in main.
diag_output = 1;
diag_fn = 0;     % Diagnostics flag, if 1, then all the functions display plots for diagnostics, Set it to 0 to avoid plots from within the calling functions
% rng(1);        % Set a common seed
No_runs = 1;    % Total number of runs to compute the rmse metric for each of the filters for comparison

%% Parameters of the target density - 2 component Gaussian mixture density 
m = 2;
sigma = [0.4 0.4]; 
mu    = [-1 1]; 
w     = [0.5 rand]; % Needs to add up to 1.
w(m)  = 1 - sum(w(1:m-1));

p = 0;         
for i = 1: length(mu)
    p = p + w(i) * exp(-(x - mu(i)).^2 / (2 * sigma(i)^2)) * (1 / sqrt(2 * pi * sigma(i)^2));
end
U      = -log(p);
grad_U = diff(U);

p_x      = matlabFunction(p);
U_x      = matlabFunction(U);
grad_U_x = matlabFunction(grad_U);

%% Function of interest 
c = @(x)x;

%% Computing the exact mean eta 
step = 0.05;
range = min(mu)- 3 * max(sigma): step : max(mu) + 3 * max(sigma);
eta = matlabFunction(sym(c) * p);
eta = sum(eta(range));

%% Flags to be set to choose the MCMC sampling method 
iid        = 0;
langevin   = 0;
metropolis = 1;
% mala     = 1;

% Sampling parameters
N  = 5000;            % No of samples obtained
gamma = 0.1;          % Time steps / variance parameter for Langevin and MH algorithms
sgamma = sqrt(gamma); % Std deviation parameter
    
%% Flags to be set to choose which approximation methods to compare
exact = 1;       % Computes the exact h' and plots 
fin   = 1;       % Computes h' using finite dimensional basis
coif  = 1;       % Computes h' using Coifman kernel method
rkhs  = 1;       % Computes h' using RKHS
zero_mean = 1;   % Computes h' using RKHS with Lagrange multipliers
const = 1;       % Computes the constant gain approximation
% Flag for variance minimization - ZV-MCMC based on Mira et al. 
var_min = 1; 

% i) Finite dimensional basis
if fin == 1
   d = 20;           % No of basis functions
   basis = 1;        % 0 for polynomial, 1 for Fourier
   p_wt =    0;        % 1 for weighting with density, 0 otherwise 
end

% ii) Coifman kernel 
if coif == 1
   eps_coif = 0.1;   % Time step parameter
   N_coif = 500;     % Number of particles/samples to place the kernel functions 
end

% iii) RKHS
if rkhs == 1
   kernel   = 0;           % 0 for Gaussian kernel
   lambda   = 1e-3;        % 0.05, 0.02, Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
   eps_rkhs = 0.25;        % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
   N_rkhs   = 500;         % Number of particles/samples where the RKHS kernel functions are computed at.
end

% iv) RKHS - zero mean
if zero_mean == 1
   kernel   = 0;           % 0 for Gaussian kernel
   lambda   = 1e-3;        % 0.05, 0.02, Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
   eps_zm   = 0.25;        % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
   N_zm     = 500;         % Number of particles/samples where the RKHS kernel functions are computed at.
end

% v) Variance minimization using finite basis
if var_min == 1
   d = 20;           % No of basis functions
   basis = 1;        % 0 for polynomial, 1 for Fourier
   p_wt =    0;        % 1 for weighting with density, 0 otherwise 
end

% Setting a max and min threshold for gain
K_max = 100;
K_min = 0;

%% Independent Monte Carlo trials
for run = 1: 1 : No_runs
    run
    common_rand = randn(N,1);
%% Sampling from the target density 
    if iid == 1
       X_iid(1) = common_rand(1);
       for n = 2:1:N
           if rand < w(1)
              X_iid(n) = mu(1) + sigma(1) * randn;
           else
              X_iid(n) = mu(2) + sigma(2) * randn;
           end
       end
       c_hat_iid(run)   = mean(c(X_iid)); 
    end

    
    if langevin == 1
       X_lang(1) = common_rand(1);
       for n = 2:1:N
           X_lang(n) = X_lang(n-1) - grad_U_x(X_lang(n-1)) * gamma +  sqrt(2) * sgamma * common_rand(n);
       end
       c_hat_lang(run)   = mean(c(X_lang)); 
    end
    
    
    if metropolis == 1
        X_mh(1) = common_rand(1);
        for n = 2:1:N
            X_mh_next = X_mh(n-1) + sgamma * common_rand(n);                   % Random Walk Metropolis chain
        % Acceptance ratio  
            alpha = p_x(X_mh_next)/p_x(X_mh(n-1));
            if(rand < alpha)
                X_mh(n) = X_mh_next;
            else
                X_mh(n) = X_mh(n-1);
            end
        end
        c_hat_mh(run)   = mean(c(X_mh)); 
    end

%% Computing the approximation \nabla h and the optimal control variates using various methods
% 0) Exact h' computation
    if exact == 1
        if iid == 1
           [K_exact_i]   = gain_exact(X_iid, c, mu, sigma, w, diag_fn );    
           [cv_X_iid]    = compute_cv(X_iid, K_exact_i, grad_U_x);
           c_exact_iid   = c(X_iid) + cv_X_iid;
           c_hat_exact_iid(run) = mean(c_exact_iid);
        end
        if langevin == 1
           [K_exact_l]   = gain_exact(X_lang, c, mu, sigma, w, diag_fn );    
           [cv_X_lang]   = compute_cv(X_lang, K_exact_l, grad_U_x);
           c_exact_lang  = c(X_lang) + cv_X_lang;
           c_hat_exact_lang(run) = mean(c_exact_lang);
        end
        if metropolis == 1
           [K_exact_m]   = gain_exact(X_mh, c, mu, sigma, w, diag_fn );    
           [cv_X_mh]     = compute_cv(X_mh, K_exact_m, grad_U_x);
           c_exact_mh    = c(X_mh) + cv_X_mh;
           c_hat_exact_mh(run) = mean(c_exact_mh);
        end
    end

% i) Finite set of basis functions of dimension d
    if fin == 1
        if iid == 1
           [K_fin_i]     = gain_fin(X_iid, c, d, basis, mu, sigma, p_wt, diag_fn );    
           [cv_X_iid]    = compute_cv(X_iid, K_fin_i, grad_U_x);
           c_fin_iid     = c(X_iid) + cv_X_iid;
           c_hat_fin_iid(run) = mean(c_fin_iid);          
        end
        if langevin == 1
           [K_fin_l]     = gain_fin(X_lang, c, d , basis, mu, sigma, p_wt, diag_fn);
           [cv_X_lang]   = compute_cv(X_lang, K_fin_l, grad_U_x);
           c_fin_lang    = c(X_lang) + cv_X_lang;
           c_hat_fin_lang(run) = mean(c_fin_lang);
        end
        if metropolis == 1
           [K_fin_m]     = gain_fin(X_mh, c, d , basis, mu, sigma, p_wt, diag_fn);
           [cv_X_mh]     = compute_cv(X_mh, K_fin_m, grad_U_x);
           c_fin_mh      = c(X_mh) + cv_X_mh;
           c_hat_fin_mh(run) = mean(c_fin_mh);
        end
    end
    
% ii) Coifman kernel semigroup approximation 
    if coif == 1
        if iid == 1
           [~, K_coif_i] = gain_coif(X_iid, c, eps_coif, [],N_coif,diag_fn );    
           [cv_X_iid]    = compute_cv(X_iid, K_coif_i, grad_U_x);
           c_coif_iid    = c(X_iid) + cv_X_iid;
           c_hat_coif_iid(run) = mean(c_coif_iid);
        end
        if langevin == 1
           [~, K_coif_l] = gain_coif(X_lang , c, eps_coif,[],N_coif, diag_fn);
           [cv_X_lang]   = compute_cv(X_lang, K_coif_l, grad_U_x);
           c_coif_lang   = c(X_lang) + cv_X_lang;
           c_hat_coif_lang(run) = mean(c_coif_lang);
        end
        if metropolis == 1
           [~,K_coif_m]  = gain_coif(X_mh , c, eps_coif, [],N_coif, diag_fn);
           [cv_X_mh]     = compute_cv(X_mh, K_coif_m, grad_U_x);
           c_coif_mh     = c(X_mh) + cv_X_mh;
           c_hat_coif_mh(run) = mean(c_coif_mh);
        end
    end 
    
% iii) RKHS based approximation 
    if rkhs == 1
        if iid == 1
           [~, K_rkhs_i] = gain_rkhs_mcmc(X_iid, c, kernel, lambda, eps_rkhs, N_rkhs, diag_fn );    
           [cv_X_iid]    = compute_cv(X_iid, K_rkhs_i, grad_U_x);
           c_rkhs_iid    = c(X_iid) + cv_X_iid;
           c_hat_rkhs_iid(run) = mean(c_rkhs_iid);
        end
        if langevin == 1
           [~, K_rkhs_l] = gain_rkhs_mcmc(X_lang , c, kernel,lambda, eps_rkhs, N_rkhs, diag_fn);
           [cv_X_lang]   = compute_cv(X_lang, K_rkhs_l, grad_U_x);
           c_rkhs_lang   = c(X_lang) + cv_X_lang;
           c_hat_rkhs_lang(run) = mean(c_rkhs_lang);
        end
        if metropolis == 1
           [~, K_rkhs_m] = gain_rkhs_mcmc(X_mh , c, kernel,lambda, eps_rkhs, N_rkhs, diag_fn);
           [cv_X_mh]     = compute_cv(X_mh, K_rkhs_m, grad_U_x);
           c_rkhs_mh     = c(X_mh) + cv_X_mh;
           c_hat_rkhs_mh(run) = mean(c_rkhs_mh);
        end    
    end
    
%    iv) RKHS - Zero mean 
    if zero_mean == 1
        if iid == 1
           [~, K_zm_i]   = gain_rkhs_zm_mcmc(X_iid', c, 1, kernel, lambda, eps_zm, N_zm, diag_fn );    
           [cv_X_iid]    = compute_cv(X_iid, K_zm_i', grad_U_x);
           c_zm_iid      = c(X_iid) + cv_X_iid;
           c_hat_zm_iid(run) = mean(c_zm_iid);
        end
        if langevin == 1
           [~, K_zm_l]   = gain_rkhs_zm_mcmc(X_lang' , c, 1, kernel,lambda, eps_zm, N_zm, diag_fn);
           [cv_X_lang]   = compute_cv(X_lang, K_zm_l', grad_U_x);
           c_zm_lang     = c(X_lang) + cv_X_lang;
           c_hat_zm_lang(run) = mean(c_zm_lang);
        end
        if metropolis == 1
           [~, K_zm_m]   = gain_rkhs_zm_mcmc(X_mh' , c, 1, kernel,lambda, eps_zm, N_zm, diag_fn);
           [cv_X_mh]     = compute_cv(X_mh, K_zm_m', grad_U_x);
           c_zm_mh       = c(X_mh) + cv_X_mh;
           c_hat_zm_mh(run) = mean(c_zm_mh);
        end    
    end
    
 % iv) Variance minimization algorithm - ZV MCMC Mira et al.     
    if var_min == 1
        if iid == 1
           [K_var_fin_i] = var_min_fin(X_iid, c, d, basis, mu, sigma, p_wt, grad_U, diag_fn);
           [cv_X_iid]    = compute_cv(X_iid, K_var_fin_i, grad_U_x);
           c_fin_iid     = c(X_iid) + cv_X_iid;
           c_var_hat_fin_iid(run) = mean(c_fin_iid);
        end
        if langevin == 1
            [K_var_fin_l] = var_min_fin(X_lang, c, d, basis, mu, sigma, p_wt, grad_U, diag_fn);
            [cv_X_iid]    = compute_cv(X_lang, K_var_fin_l, grad_U_x);
            c_fin_lang    = c(X_lang) + cv_X_lang;
            c_var_hat_fin_lang(run) = mean(c_fin_lang);
        end
        if metropolis == 1
            [K_var_fin_m] = var_min_fin(X_mh, c, d, basis, mu, sigma, p_wt, grad_U, diag_fn);
            [cv_X_mh]     = compute_cv(X_mh, K_var_fin_m, grad_U_x);
            c_fin_mh      = c(X_mh) + cv_X_mh;
            c_var_hat_fin_mh(run) = mean(c_fin_mh);
        end      
    end

% vi) Constant gain approx - Using constant h'
    if const == 1
        if iid == 1
           K_const_i     = mean((c(X_iid) - c_hat_iid(run)) .* X_iid) * ones(1,N);
           [cv_X_iid]    = compute_cv(X_iid, K_const_i, grad_U_x);
           c_const_iid   = c(X_iid) + cv_X_iid;   
           c_hat_const_iid(run) = mean(c_const_iid);
        end
        if langevin == 1
           K_const_l     = mean((c(X_lang) - c_hat_lang(run)) .* X_lang) * ones(1,N);
           [cv_X_lang]   = compute_cv(X_lang, K_const_l, grad_U_x);
           c_const_lang  = c(X_lang) + cv_X_lang;
           c_hat_const_lang(run) = mean(c_const_lang);
        end
        if metropolis == 1
           K_const_m     = mean((c(X_mh) - c_hat_mh(run)) .* X_mh) * ones(1,N);
           [cv_X_mh]     = compute_cv(X_mh, K_const_m, grad_U_x);
           c_const_mh    = c(X_mh) + cv_X_mh;
           c_hat_const_mh(run) = mean(c_const_mh);
        end
    end

    
     
%% Displaying figures for diagnostics 
    if ( diag_main == 1 && No_runs == 1) 
        if iid == 1
            figure;
            if exact == 1
                plot(X_iid, K_exact_i, 'rv','DisplayName','Exact'); 
                hold on;
            end
            if fin == 1
                plot(X_iid, K_fin_i, 'g*', 'DisplayName','Finite');  
                hold on;
            end
            if coif == 1
                plot(X_iid, K_coif_i, 'bo','DisplayName','Coifman');
                hold on;
            end
            if rkhs == 1
                plot(X_iid, K_rkhs_i, 'k^','DisplayName','RKHS');
                hold on;
            end
            if zero_mean == 1
                plot(X_iid, K_zm_i, 'cs','DisplayName','RKHS(ZM)');
                hold on;
            end
            if const ==1
                plot(X_iid, K_const_i,'mv','DisplayName','Const');
                hold on;
            end
            title(['\nabla h at particle locations for ' num2str(N) ' iid samples']);
            legend('show');
        end
        
        if langevin == 1
            figure;
            if exact == 1
                plot(X_lang, K_exact_l, 'rv','DisplayName','Exact'); 
                hold on;
            end
            if fin == 1
                plot(X_lang, K_fin_l, 'g*', 'DisplayName','Finite');  
                hold on;
            end
            if coif == 1
                plot(X_lang, K_coif_l, 'bo','DisplayName','Coifman');
                hold on;
            end
            if rkhs == 1
                plot(X_lang, K_rkhs_l, 'k^','DisplayName','RKHS');
                hold on;
            end
            if zero_mean == 1
                plot(X_lang, K_zm_l, 'cs','DisplayName','RKHS(ZM)');
                hold on;
            end
            if const ==1
                plot(X_lang, K_const_l,'mv','DisplayName','Const');
                hold on;
            end
            title(['\nabla h at particle locations for ' num2str(N) ' samples from Langevin diffusion']);
            legend('show');
        end
        
        if metropolis == 1
            figure;
            if exact == 1
                plot(X_mh, K_exact_m, 'rv','DisplayName','Exact'); 
                hold on;
            end
            if fin == 1
                plot(X_mh, K_fin_m, 'g*', 'DisplayName','Finite');  
                hold on;
            end
            if coif == 1
                plot(X_mh, K_coif_m, 'bo','DisplayName','Coifman');
                hold on;
            end
            if rkhs == 1
                plot(X_mh, K_rkhs_m, 'k^','DisplayName','RKHS');
                hold on;
            end
            if zero_mean == 1
                plot(X_mh, K_zm_m, 'cs','DisplayName','RKHS(ZM)');
                hold on;
            end
            if const ==1
                plot(X_mh, K_const_m,'mv','DisplayName','Const');
                hold on;
            end
            title(['\nabla h at particle locations for ' num2str(N) 'samples from Metropolis-Hastings']);
            legend('show');
        end
          
       figure(100);
       plot(range,p_x(range),'k','linewidth',2.0,'DisplayName','Target density - \rho');
       hold on;
       v = version('-release');
       if (v == '2014a')
           if iid == 1
               hist(X_iid,N);
            end
            if langevin == 1
               hist(X_lang,N);
            end
            if metropolis == 1
               hist(X_mh,N);
            end          
       else
            if iid == 1
             % CAUTION : histogram command works only in recent Matlab
             % versions, if it does not work, comment this section out
               histogram(X_iid,'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'DisplayName','iid');
               hold on;
            end
            if langevin == 1
               histogram(X_lang,'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'DisplayName','Langevin');
               hold on;
            end
            if metropolis == 1
               histogram(X_mh,'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'DisplayName','RWM');
               hold on;
            end
       end  
       legend('show');
       title('Target density \rho and Histogram of samples obtained');
    end
end

%% Computing the mean across independent trials
if No_runs > 1
if iid == 1
   c_overall_iid  = mean(c_hat_iid);
   asym_var_iid   = var(sqrt(N) * (c_hat_iid - eta));
   if exact == 1
      c_overall_exact_iid = mean(c_hat_exact_iid);
      asym_var_exact_iid  = var(sqrt(N) * (c_hat_exact_iid - eta));
   end
   if fin == 1
      c_overall_fin_iid = mean(c_hat_fin_iid);
      asym_var_fin_iid  = var(sqrt(N) * (c_hat_fin_iid - eta));      
   end
   if coif == 1
      c_overall_coif_iid = mean(c_hat_coif_iid);
      asym_var_coif_iid  = var(sqrt(N) * (c_hat_coif_iid - eta));
   end
   if rkhs == 1
      c_overall_rkhs_iid = mean(c_hat_rkhs_iid);
      asym_var_rkhs_iid  = var(sqrt(N) * (c_hat_rkhs_iid - eta));
   end
   if zero_mean == 1
      c_overall_zm_iid = mean(c_hat_zm_iid);
      asym_var_zm_iid  = var(sqrt(N) * (c_hat_zm_iid - eta));
   end
   if var_min == 1
      c_overall_var_fin_iid = mean(c_var_hat_fin_iid);
      var_fin_iid           = var(sqrt(N) * (c_var_hat_fin_iid - eta));
   end
   if const == 1
      c_overall_const_iid = mean(c_hat_const_iid);
      asym_var_const_iid  = var(sqrt(N) * (c_hat_const_iid - eta));
   end
end
   
if langevin == 1
   c_overall_lang  = mean(c_hat_lang);
   asym_var_lang   = var(sqrt(N) * (c_hat_lang - eta));
   if exact == 1
      c_overall_exact_lang = mean(c_hat_exact_lang);
      asym_var_exact_lang  = var(sqrt(N) * (c_hat_exact_lang - eta));
   end
   if fin == 1
      c_overall_fin_lang = mean(c_hat_fin_lang);
      asym_var_fin_lang  = var(sqrt(N) * (c_hat_fin_lang - eta));
   end
   if coif == 1
      c_overall_coif_lang = mean(c_hat_coif_lang);
      asym_var_coif_lang  = var(sqrt(N) * (c_hat_coif_lang - eta));
   end
   if rkhs == 1
      c_overall_rkhs_lang = mean(c_hat_rkhs_lang);
      asym_var_rkhs_lang  = var(sqrt(N) * (c_hat_rkhs_lang - eta));
   end
   if zero_mean == 1
      c_overall_zm_lang = mean(c_hat_zm_lang);
      asym_var_zm_lang  = var(sqrt(N) * (c_hat_zm_lang - eta));
   end
   if var_min == 1
      c_overall_var_fin_lang = mean(c_var_hat_fin_lang);
      var_fin_lang           = var(sqrt(N) * (c_var_hat_fin_lang - eta));
   end
   if const == 1
      c_overall_const_lang = mean(c_hat_const_lang);
      asym_var_const_lang  = var(sqrt(N) * (c_hat_const_lang - eta));
   end
end

if metropolis == 1
   c_overall_mh  = mean(c_hat_mh);
   asym_var_mh   = var(sqrt(N) * (c_hat_mh - eta));
   if exact == 1
      c_overall_exact_mh = mean(c_hat_exact_mh);
      asym_var_exact_mh  = var(sqrt(N) * (c_hat_exact_mh - eta));
   end
   if fin == 1
      c_overall_fin_mh = mean(c_hat_fin_mh);
      asym_var_fin_mh  = var(sqrt(N) * (c_hat_fin_mh - eta));
   end
   if coif == 1
      c_overall_coif_mh = mean(c_hat_coif_mh);
      asym_var_coif_mh  = var(sqrt(N) * (c_hat_coif_mh - eta));
   end
   if rkhs == 1
      c_overall_rkhs_mh = mean(c_hat_rkhs_mh);
      asym_var_rkhs_mh  = var(sqrt(N) * (c_hat_rkhs_mh - eta));
   end
   if zero_mean == 1
      c_overall_zm_mh = mean(c_hat_zm_mh);
      asym_var_zm_mh  = var(sqrt(N) * (c_hat_zm_mh - eta));
   end
   if var_min == 1
      c_overall_var_fin_mh = mean(c_var_hat_fin_mh);
      var_fin_mh           = var(sqrt(N) * (c_var_hat_fin_mh - eta));
   end
   if const == 1
      c_overall_const_mh = mean(c_hat_const_mh);
      asym_var_const_mh  = var(sqrt(N) * (c_hat_const_mh - eta));
   end
end

% Histogram of asymptotic variance 
style = 'bar';         % or stairs
bw    = 0.1;           % bin width
figure;
if iid == 1
   histogram(sqrt(N) * (c_hat_iid - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Standard MC');
   hold on;
   if exact == 1
      histogram(sqrt(N) * (c_hat_exact_iid - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Exact');
      hold on;
   end
   if fin == 1
      histogram(sqrt(N) * (c_hat_fin_iid - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Finite');
      hold on;
   end
   if var_min == 1
      histogram(sqrt(N) * (c_var_hat_fin_iid - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Var. min');
      hold on;
   end
   if coif == 1
      histogram(sqrt(N) * (c_hat_coif_iid - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Coifman');
      hold on;
   end
   if rkhs == 1
      histogram(sqrt(N) * (c_hat_rkhs_iid - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','RKHS');
      hold on;
   end
   if zero_mean == 1
      histogram(sqrt(N) * (c_hat_zm_iid - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','RKHS(ZM)');
      hold on;
   end
   if const == 1
      histogram(sqrt(N) * (c_hat_const_iid - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Const');
      hold on;
   end
   title(['Asymptotic variance over ' num2str(No_runs) ' independent trials using iid sampling']); 
   legend('show');
end

figure;
if langevin == 1
   histogram(sqrt(N) * (c_hat_lang - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Standard MC');
   hold on;
   if exact == 1
      histogram(sqrt(N) * (c_hat_exact_lang -eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Exact');
      hold on;
   end
   if fin == 1
      histogram(sqrt(N) * (c_hat_fin_lang - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Finite');
      hold on;
   end
   if var_min == 1
      histogram(sqrt(N) * (c_var_hat_fin_lang - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Var. min');
      hold on;
   end
   if coif == 1
      histogram(sqrt(N) * (c_hat_coif_lang - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Coifman');
      hold on;
   end
   if rkhs == 1
      histogram(sqrt(N) * (c_hat_rkhs_lang - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','RKHS');
      hold on;
   end
   if zero_mean == 1
      histogram(sqrt(N) * (c_hat_zm_lang - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','RKHS(ZM)');
      hold on;
   end
   if const == 1
      histogram(sqrt(N) * (c_hat_const_lang - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Const');
      hold on;
   end
   title(['Asymptotic variance over ' num2str(No_runs) ' independent trials using Langevin sampling']); 
   legend('show');
end

figure;
if metropolis == 1
   histogram(sqrt(N) * (c_hat_mh - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Standard MC');
   hold on;
   if exact == 1
      histogram(sqrt(N) * (c_hat_exact_mh - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Exact');
      hold on;
   end
   if fin == 1
      histogram(sqrt(N) * (c_hat_fin_mh - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Finite');
      hold on;
   end
   if var_min == 1
      histogram(sqrt(N) * (c_var_hat_fin_mh - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Var. min');
      hold on;
   end
   if coif == 1
      histogram(sqrt(N) * (c_hat_coif_mh - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Coifman');
      hold on;
   end
   if rkhs == 1
      histogram(sqrt(N) * (c_hat_rkhs_lang - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','RKHS');
      hold on;
   end
   if zero_mean == 1
      histogram(sqrt(N) * (c_hat_zm_lang - eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','RKHS(ZM)');
      hold on;
   end
   if const == 1
      histogram(sqrt(N) * (c_hat_const_mh -eta),'Normalization','pdf','DisplayStyle',style,'BinWidth',bw,'DisplayName','Const');
      hold on;
   end
   title(['Asymptotic variance over ' num2str(No_runs) ' independent trials using RWM sampling']); 
   legend('show');
end
end






