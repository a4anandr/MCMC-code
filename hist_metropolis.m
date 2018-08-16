%% Main function to plot the reduction in asymptotic variance with increase in basis dimension for RWM
% Also plots histograms of mean estimates

clc;
clear all;
close all;

basis   = 0;               % 0 for polynomial, 1 for Fourier - both weighted with Gaussian densities
diag_fn = 0;               % 0 for No display, 1 to display plots from called functions
diag_main = 1;             % 0 for No display, 1 to display plots within the main function
tic;

syms x;

% State space domain
xmin = -3;
xmax = 3;
step = 0.01;
X=xmin:step:xmax;

% Time domain specifications
N = 10000;           % Total number of samples obtained from RWM
gamma = 0.1;          % Step size parameter used in RWM
burnin = 2;           % Number of samples burned in 

% For histogram
No_trials =100;     % Total number of independent trials

% Subexponential density
delta = 1;    % 0 < delta < 1, delta = 1 => Gaussian 

%% Defining the parameters of the target density - for RWM
mu    = [-1 1];
sig   = [ 0.4 0.4];
w     = [ 0.5 0.5];

p = 0;
for run = 1 : length(mu)
    p   = p + w(run) * exp(- (norm(x - mu(run))/( 2 * sig(run)))^(2 * delta));   % unnormalized density ;
end   
p_x  = matlabFunction(p);

% Defining potential function U(x)
U        = -log(p);
grad_U   = diff(U);
grad_U_x = matlabFunction(grad_U);

% Function whose mean is to be computed
c = x;    % c = sin(x);
c_x = matlabFunction(c);

cp  = matlabFunction(c*p);
eta = sum(cp(X) * step);

%% Computation of the exact solution to Poisson's equation - K_act, This will be used to come up with the theoretical value of asymptotic variance, 2 <K, K>.
[K]      = gain_exact(X, c_x, mu, sig, w, diag_fn); 
gamma2_0 = 2 * sum(K .* K .* p_x(X) * step);

%% Basis Functions
gamma2_d = gamma2_0;
l = [6 10 20];          % List the basis dimensions you want to plot 

for d = 1 : 1 : length(l) 
    % d = 20;
    l(d)
    [K_theta c_theta]  = gain_and_cv_fin(X, c_x, l(d), basis, p, mu, sig, 1, 2, diag_fn);
    K_theta_x = matlabFunction(K_theta);
    c_theta_x = matlabFunction(c_theta);

% Theoretical variance with control variates
    gamma2_d(d + 1) = 2 * sum((K - K_theta_x(X)).*(K - K_theta_x(X)).*p_x(X) * step);
   
%% Metropolis Hastings algorithm for sampling
% RWM with step-size parameter \gamma 

    X_mh = randn;                
    c_hat_mh_cv(1)  = c_theta_x(X_mh);
    c_hat_mh(1)     = c_x(X_mh);
    for run = 1:1:No_trials
        run  
        for k = 1:1:length(gamma)
            for n = 1: 1 : N
                X_mh_next = X_mh(n) + sqrt(2 * gamma(k)) * randn;         % Possible next sample obtained using RW from the current sample Phi(t)
   % Acceptance ratio  
                alpha = p_x(X_mh_next)/p_x(X_mh(n));
                if(rand < alpha)
                    X_mh(n+1) = X_mh_next;
                else
                    X_mh(n+1) = X_mh(n);
                end
            % c_hat_mh_cv(n+1) = (n/(n+1)) * c_hat_mh_cv(n) + (1/(n+1)) * c_theta_x(X_mh(n+1));  
            % c_hat_mh(n+1)    = (n/(n+1)) * c_hat_mh(n)  + (1/(n+1)) * c_x(X_mh(n+1));  
           end
   % Monte Carlo estimate for M and b
          [K_theta_2] = gain_fin(X_mh(burnin+1 : end),c_x,l(d),basis,mu,sig,1,2,diag_fn); 
          [cv_X_mh]   = compute_cv(X_mh(burnin+1 : end), K_theta_2, grad_U_x);

          c_hat_mh(run)        = mean(c_x(X_mh(burnin+1 : end)));
          c_hat_mh_cv(run)     = mean(c_theta_x(X_mh(burnin+1 : end)));  
          c_hat_mh_cv_2(run)   = mean(c_x(X_mh(burnin+1 : end)) + cv_X_mh);
        end
    end

%% Histogram and asymptotic variance computation
    eta_hat      = mean(c_hat_mh)
    eta_hat_lang = mean(c_hat_mh_cv)
    eta_hat_lang2= mean(c_hat_mh_cv_2)

    hist_variance(d,:)     = sqrt((N - burnin))*(c_hat_mh_cv - eta);
    hist_variance_0(d,:)   = sqrt((N - burnin))*(c_hat_mh - eta);
    hist_variance_2(d,:)   = sqrt((N - burnin))*(c_hat_mh_cv_2 - eta);

%% Figures
% Histogram of the RWM
if diag_main == 1 && No_trials > 1
  v = version('-release');
       if (v == '2014a')
          [dist,loc]= hist(X_mh, 0.01 * N);
          figure;   
          hist(X_mh, 0.01* N);
          hold on;
          plot(X,p_x(X)*(max(max(dist))/max(p_x(X))),'r','linewidth',2.0);
          title('Histogram of the particles with density p');
          
          figure;
          span = ceil(max(hist_variance(d,:)))-floor(min(hist_variance(d,:)));
          spacing = 0.1;
          bins = span/spacing;
          [weight, loc] = hist(hist_variance(d,:),bins);
          histfit(hist_variance(d,:),bins);
          hold on;
          title(['Histogram of \eta^{i} - \eta for ' num2str(No_trials) ' independent trials']);
          legend('Histogram','Gaussian fit');
          
      % Histogram for d = 0
          figure;
          span_0 = ceil(max(hist_variance_0(d,:)))-floor(min(hist_variance_0(d,:)));
          spacing_0 = 0.1;
          bins_0 = span_0/spacing_0;
          [weight, loc_0] = hist(hist_variance_0(d,:),bins_0);
          histfit(hist_variance_0(d,:),bins_0);
          hold on;
          title(['Histogram of \eta^{i} - \eta for ' num2str(N) 'independent trials']);
          legend('Histogram','Gaussian fit');
    % save(['hist_d_0_' num2str(d) 'N_' num2str(N) 'T_' num2str(T) '.mat'],'hist_variance_0','sigmahat_0');

          figure;
          hist(hist_variance_0(d,:),bins_0);
          h = findobj(gca,'Type','patch');
          h.FaceColor = 'r';
          hold on;
          hist(hist_variance(d,:),bins);
          legend('Standard MC',['l=' num2str(l(d))]);
          title('Langevin');
       else 
          figure;
          histogram(X_mh,'Normalization','pdf','DisplayStyle','bar','BinWidth',0.01,'DisplayName','Samples');
          hold on;
          plot(X,p_x(X),'r');
          legend('show');
          
          figure;
          histogram(hist_variance_0(d,:),'Normalization','pdf','DisplayStyle','bar','BinWidth',1,'DisplayName','Standard MC');
          hold on;
          histogram(hist_variance(d,:),'Normalization','pdf','DisplayStyle','bar','BinWidth',1,'DisplayName',['l = ' num2str(l(d)) ' Integration']);
          histogram(hist_variance_2(d,:),'Normalization','pdf','DisplayStyle','bar','BinWidth',1,'DisplayName',['l = ' num2str(l(d)) ' Empirical']);
          title(['Histogram of \eta^{i} - \eta for ' num2str(No_trials) 'independent trials']);
          legend('show');   
       end
    end
toc
end

