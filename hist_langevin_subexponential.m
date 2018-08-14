%% Main function to plot the reduction in asymptotic variance with increase in basis dimension for Langevin diffusion
% Also plots histograms of mean estimates 
clc;
clear all;
close all;

basis   = 0;               % 0 for polynomial, 1 for Fourier - both weighted with Gaussian densities
diag_fn = 0;               % 0 for No display, 1 for display
diag_main = 1;
tic;

syms x

% State space domain
xmin = -3;
xmax = 3;
step = 0.01;
X = xmin : step : xmax;

% Time domain specifications
N     = 100000;
dt    = 0.1;
sdt   = sqrt(dt);
sqrt2 = sqrt(2);

% For histogram
No_trials   = 100;

% Subexponential density
delta     = 1;    % 0 < delta < 1, delta = 1 => Gaussian 

%% Defining the parameters of the target density - U and sigmaB for Langevin diffusion
mu    = [-1 1];
sig   = [ 0.4 0.4];
w     = [ 0.5 0.5];

p = 0;
for i = 1 : length(mu)
    p   = p + w(i) * exp(- (norm(x - mu(i))/( 2 * sig(i)))^(2 * delta));   % unnormalized density ;
end   
p_x  = matlabFunction(p);

% Defining potential function U(x)
U        = -log(p);
grad_U   = diff(U);
grad_U_x = matlabFunction(grad_U);

% Observation function
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
    
%% Langevin diffusion for sampling
    X_lang = randn;
    c_hat_lang_cv(1) = c_theta_x(X_lang);
    c_hat_lang(1)      = c_x(X_lang);
    for run = 1:1:No_trials
        run
        rng(run + No_trials)
        for n= 1 : 1 : N
            X_lang(n+1) = X_lang(n) - grad_U_x(X_lang(n))*dt +  sqrt2 * sdt * randn;
            % c_hat_lang_cv(n+1) = (n/(n+1)) * c_hat_lang_cv(n) + (1/(n+1)) * c_theta_x(X_lang(n+1));  
            % c_hat_lang(n+1)    = (n/(n+1)) * c_hat_lang(n)  + (1/(n+1)) * c_x(X_lang(n+1));  
        end
        c_hat_lang_cv(run)   = mean(c_theta_x(X_lang));
        c_hat_lang(run)      = mean(c_x(X_lang));
        % c_hat_lang_cv(run) = c_hat_lang_cv(end);
        % c_hat_lang(run)    = c_hat_lang(end);
    end

%% Histogram and asymptotic variance computation
    hist_variance(d,:)   = sqrt(N)*(c_hat_lang_cv - eta);
    hist_variance_0(d,:) = sqrt(N)*(c_hat_lang - eta);

%% Figures
    if diag_main == 1 && No_trials > 1
    % Histogram of the Langevin diffusion particles
       v = version('-release');
       if (v == '2014a')
          [dist,loc]= hist(X_lang, 0.01 * N);
          figure;   
          hist(X_lang, 0.01* N);
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
          title(['Histogram of \eta^{i} - \eta for N =' num2str(N)]);
          legend('Histogram','Gaussian fit');
          
      % Histogram for d = 0
          figure;
          span_0 = ceil(max(hist_variance_0(d,:)))-floor(min(hist_variance_0(d,:)));
          spacing_0 = 0.1;
          bins_0 = span_0/spacing_0;
          [weight, loc_0] = hist(hist_variance_0(d,:),bins_0);
          histfit(hist_variance_0(d,:),bins_0);
          hold on;
          title(['Histogram of \eta^{i} - \eta for N =' num2str(N)]);
          legend('Histogram','Gaussian fit');
    % save(['hist_d_0_' num2str(d) 'N_' num2str(N) 'T_' num2str(T) '.mat'],'hist_variance_0','sigmahat_0');

          figure;
          hist(hist_variance_0(d,:),bins_0);
          h = findobj(gca,'Type','patch');
          h.FaceColor = 'r';
          hold on;
          hist(hist_variance(d,:),bins);
          legend('Standard MC',['l=' num2str(d)]);
          title('Langevin');
       else 
          figure;
          histogram(X_lang,'Normalization','pdf','DisplayStyle','bar','BinWidth',0.01,'DisplayName','Samples');
          hold on;
          plot(X,p_x(X),'r');
          legend('show');
          
          figure;
          histogram(hist_variance_0(d,:),'Normalization','pdf','DisplayStyle','bar','BinWidth',1,'DisplayName','Standard MC');
          hold on;
          histogram(hist_variance(d,:),'Normalization','pdf','DisplayStyle','bar','BinWidth',1,'DisplayName',['l = ' num2str(d)]);
          title(['Histogram of \eta^{i} - \eta for ' num2str(No_trials) 'independent trials']);
          legend('show');   
       end
    end
toc
end





