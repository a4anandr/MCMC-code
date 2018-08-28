function [eta K] = gain_rkhs_zm_mcmc( Xi , c , d, kernel, lambda, epsilon, N_ker, diag)
% Returns the gain computed at particle locations Xi using an RKHS 
tic;
N = length(Xi);
rand_train = 0;                        % For now set this to 0; rand_train = 1 needs fixing
if rand_train ==1
%    Xi_train = randsample(Xi,N_ker);
    ind_train = randperm(N,N_ker);
else
    i = 1;
    i_Nker = 1;
    ind_train = 1;
    while i_Nker < N_ker
        if Xi(ind_train) ~= Xi(i+1) 
            ind_train(i_Nker+1) = i + 1;
            i = i + 1;
            i_Nker = i_Nker + 1;
        else
            i = i + 1;
            continue;
        end
    end
end
Xi_train  = Xi(ind_train);

% Evaluation of kernel matrices 
v = version('-release');
if kernel == 0  % Gaussian
    if v == '2018a'
        for k = 1 : N_ker
            Ker(:,k)   = exp(- vecnorm((Xi - Xi_train(k) * ones(N,1)),2,2).^2 / (4 * epsilon));
            for d_i = 1 : d
                Ker_x(:,k,d_i) = - (Xi(:,d_i) - Xi_train(k,d_i) * ones(N,1))./ (2 * epsilon) .* Ker(:,k);
            end
        end
    else
        for i = 1:N
           for k = 1 : N_ker          
               if (Xi(i) ~= Xi(k)) || (i == k)
                   Ker(i,k) =  exp(-(norm(Xi(i) - Xi_train(k)).^2/(4 * epsilon)));    
                   for d_i = 1 : d
                       Ker_x(i,k,d_i) =  - (Xi(i,d_i) - Xi_train(k,d_i)) / (2 * epsilon) .* Ker(i,k);
                   end
               end
           end
       end
    end  
   
H     = c(Xi);       
eta   = mean(c(Xi));
Y     =  (H - eta)';
% for i = 1:N
%     K_hat = K_hat + (1/N) * ((c(Xi(i,:)) - eta) .* Xi(i,:));      % Constant gain approximation
% end

K_hat = mean( (c(Xi) - eta) .* Xi);
% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
Ker_x_sum = zeros(N_ker,N_ker);
for d_i = 1 : d
    Ker_x_ones(:,d_i) =  Ker_x(:,:,d_i) * ones(N_ker,1);
    Ker_x_sum         =  Ker_x_sum + Ker_x(:,:,d_i)'* Ker_x(:,:,d_i);
    M(1 : N_ker, N_ker + d_i) =  Ker_x_ones(ind_train, d_i);        % Ker_x_ones(1 : N_ker, d_i)
    M(N_ker + d_i, 1 : N_ker) =  Ker_x_ones(ind_train, d_i)';
end
b(1 : N_ker, :)            =  (2/N) * Ker' * Y'- (2/N) * Ker_x_ones (ind_train) * K_hat'; % (2/N) * Ker' * Y'- (2/N) * Ker_x_ones (1:N_ker) * K_hat'; 
b(N_ker+1 : N_ker+d, :)    =  zeros(d,1);    
M(1 : N_ker, 1 : N_ker)    =  2 * lambda * Ker (ind_train, :) + ( 2 / N) * Ker_x_sum;       % Ker_x * Ker_x' = Ker_x' * Ker_x - Hence either one works
beta               =  M \ b;
        
K  = K_hat * ones(N,1);
for k = 1 : 1 : N_ker
    for d_i = 1 : d
        K(:,d_i)      = K(:,d_i)     + beta(k)  * Ker_x(:,k,d_i);      % Ker_x(pj,pi)
    end           
end

toc
 
%% For displaying figures
if diag == 1
    figure;
    plot(Xi,K,'b*');    
end
  
end


