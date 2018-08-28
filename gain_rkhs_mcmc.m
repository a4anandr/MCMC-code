function [beta K] = gain_rkhs( Xi , c , kernel, lambda, epsilon, N_ker, diag)
% Returns the gain computed at particle locations Xi using an RKHS 
tic 
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
if kernel == 0
    if v == '2018a'
        for k = 1:N_ker          % Need to fix this for RWM where Xi_train could have repetitions of the same sample
            % Following needs to be computed only on diagonal elements,
            Ker(:,k)   = exp(- vecnorm((Xi' - Xi_train(k) * ones(N,1)),2,2).^2 / (4 * epsilon));
            Ker_x(:,k) = - (Xi' - Xi_train(k) * ones(N,1))./ (2 * epsilon) .* Ker(:,k);              
        end

    else   % older Matlab versions that do not have vecnorm
        for i = 1:N
            for k = 1:N_ker          
                if (Xi(i) ~= Xi_train(k)) || (i == ind_train(k))
                   Ker(i,k)   =  exp(-(norm(Xi(i) - Xi_train(k)).^2/(4 * epsilon)));  
                   Ker_x(i,k) =  - (Xi(i) - Xi_train(k)) / (2 * epsilon) .* Ker(i,k);
                end
            end
        end 
    end
end
   
 H     = c(Xi);       
 eta   = mean(c(Xi));
 Y     =  (H - eta)';
% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
 b     = (1/N) * ( Ker' * Y); 
 M     = lambda * Ker(ind_train,:)' + (1 / N) * Ker_x' * Ker_x;       % Ker_x * Ker_x' = Ker_x' * Ker_x - Hence either one works
 beta  = M \ b;
 
K = zeros(1,N);
for k = 1 : 1 : N_ker
    K      = K   + beta(k)  * Ker_x(:,k)';      % Ker_x(pj,pi)
end

toc
 
%% For displaying figures
if diag == 1
    figure;
    plot(Xi,K,'b*');
    
    figure;
    hist(Xi_train,ceil(N_ker/10));
end
  
end

