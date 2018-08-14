
clc;
close all;
clear;

No_trials = 1;

%% Draw random numbers
rng('shuffle', 'twister');

%% Load and prepare data
load('./examples/logitNormalPriorSwiss/data/swiss.mat');
y = data(:,end);
data(:, end) = [];
[n, d] = size(data);
        
%% Standardise data
data = (data-repmat(mean(data), n, 1))./repmat(std(data), n, 1);
        
%% Create polynomial basis
polynomialOrder = 1;
X = zeros(n, d*polynomialOrder);
for i = 1:polynomialOrder
  X(:, ((i-1)*d+1):i*d) = data.^i;
end

%% Run logitNormalPriorSwissMetropolis
for trial = 1 : 1 : No_trials    
    trial
    model = LogitNormalPrior(X, y, 100);     % Class named LogitNormalPrior - It has a number of functions to compute logposterior, grad log posterior etc.

    options.initialParameters = zeros(1, model.np);

    options.nMcmc = 100000;
    options.nBurnIn = 10000;
    options.proposalWidthCorrection = 0.1;
    options.monitorRate = 100;

    metropolisOutput = metropolis(model, options);
    B = metropolisOutput{1};
    Z = metropolisOutput{2};
    metropolisParameters = metropolisOutput{3};

%% Compute ZV-RMHMC estimates based on linear polynomial
      [BZvL polCoefL]   = linearZv(B, Z);
      [BZvQ polCoefQ]   = quadraticZv(B, Z);

%% Compute the new theta's based on the asymptotic variance minimizing control variate 

     [BZavL polCoefaL] = linearav(B, Z);
     [BZavQ polCoefaQ] = quadraticAZv(B, Z);
     [BZavQ2 polCoefaQ2] = quadraticAv_cov(B, Z);
     % [BZavKcov polCoefavKcov]  = kernelAv_cov( B, Z);
     [BZavK polCoefavK]  = kernelAv_cov_fast( B, Z);
    
%% Save the trial averages and variances 
      aver_no_cv(trial,:)     = mean(B);
      aver_var_Q(trial,:)     = mean(BZvQ);
      aver_var_L(trial,:)     = mean(BZvL); 
      aver_asy_var_Q(trial,:) = mean(BZavQ);
      aver_asy_var_L(trial,:) = mean(BZavL);
      aver_asy_var_K(trial,:) = mean(BZavK);
%      aver_asy_var_K_cov(trial,:) = mean(BZavKcov);
      
     var_no_cv(trial,:)        = var(B);
     var_var_Q(trial,:)        = var(BZvQ);
     var_var_L(trial,:)        = var(BZvL);
     var_asy_var_Q(trial,:)    = var(BZavQ);
     var_asy_var_L(trial,:)    = var(BZavL);
     var_asy_var_K(trial,:)    = var(BZavK);
%     var_asy_var_K_cov(trial,:)    = var(BZavKcov);
end

 
 
%% Computing asymptotic variances, plotting histograms and boxplots 
if No_trials > 1
    
    save(['./examples/logitNormalPriorSwiss/output/' ...
     'averages.variances.kernel.' ...
     'nMcmc' num2str(metropolisParameters(1)) '.' ...
     'nBurnIn' num2str(metropolisParameters(2)) '.' ...
     'widthCorrection' num2str(metropolisParameters(3)) '.' ...
     'trial' num2str(No_trials) '.mat'], ...
     'aver_no_cv','aver_var_L','aver_asy_var_L','aver_var_Q','aver_asy_var_Q','var_no_cv','var_var_L','var_asy_var_L','var_var_Q','var_asy_var_Q')
    
 for i = 1:1:d
        figure;
    % hist(aver_no_cv(:,i),50);    
%     hist(aver_asy_var_L(:,i));
%     hold on;
%     hist(aver_asy_var_Q(:,i));
%     hist(aver_asy_var_K(:,i));
        no_cv_width   = max(aver_no_cv(:,i)) - min(aver_no_cv(:,i));
        asy_var_width = max(aver_asy_var_Q(:,i)) - min(aver_asy_var_Q(:,i));
        var_width     = max(aver_var_Q(:,i)) - min(aver_var_Q(:,i));    
        histogram(aver_no_cv(:,i),'Normalization','pdf','DisplayStyle','stairs','BinWidth',no_cv_width,'DisplayName','Standard MC');
        hold on;
        histogram(aver_asy_var_L(:,i),'Normalization','pdf','DisplayStyle','bar','BinWidth',asy_var_width,'DisplayName','Diff TD - Linear');
        histogram(aver_asy_var_Q(:,i),'Normalization','pdf','DisplayStyle','bar','BinWidth',asy_var_width,'DisplayName','Diff TD - Quadratic');
        histogram(aver_var_Q(:,i),'Normalization','pdf','DisplayStyle','stairs','BinWidth',var_width,'DisplayName','ZV MCMC - Linear');
        histogram(aver_var_Q(:,i),'Normalization','pdf','DisplayStyle','stairs','BinWidth',var_width,'DisplayName','ZV MCMC - Quadratic');
        histogram(aver_asy_var_K(:,i),'Normalization','pdf','DisplayStyle','stairs','BinWidth',var_width,'DisplayName','RKHS');
        legend('show');
        title(['Histogram of the regression coefficient estimate \Theta_',num2str(i)]);
    
        For_box = [aver_asy_pruned_K(:,i) aver_var_Q(1:763,i) aver_asy_var_Q(1:763,i) aver_var_L(1:763,i) aver_asy_var_L(1:763,i) aver_no_cv(1:763,i)  ];
        colors = [1 0 0; 0 1 0; 0 0 1; 0.5 0.5 0; 0.5 0 0.5; 0 0.5 0.5];
        figure;
        boxplot(For_box,'colors',colors);
        hLegend = legend(findall(gca,'Tag','Box'), {'Standard MC','Diff TD - linear','ZV - linear','Diff TD - quadratic', 'ZV - quadratic','Diff TD - RKHS'});
    end
end
