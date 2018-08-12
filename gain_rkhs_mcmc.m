function [beta_m K] = gain_rkhs( Xi , c , kernel, lambda, epsilon, N_ker, diag)
% Returns the gain computed at particle locations Xi using an RKHS 
tic 
N = length(Xi);
rand_train = 0;                        % For now set this to 0; rand_train = 1 needs fixing
if rand_train ==1
%    Xi_train = randsample(Xi,N_ker);
    ind_train = randperm(N,N_ker);
else
    ind_train = 1:N_ker;
end
Xi_train  = Xi(ind_train);

% Evaluation of kernel matrices 
if kernel == 0  % Gaussian
    for i = 1:N
       for k = 1:N_ker          
           if (Xi(i) ~= Xi_train(k)) || (i == ind_train(k))
               Ker(i,k)   =  exp(-(norm(Xi(i) - Xi_train(k)).^2/(4 * epsilon)));  
               Ker_x(i,k) =  - (Xi(i) - Xi_train(k)) / (2 * epsilon) .* Ker(i,k);
           end
       end
    end 
end  
   
 H     = c(Xi);       
 eta   = mean(c(Xi));
 Y     =  (H - eta)';
% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
 b_m     = (1/N) * ( Ker' * Y); 
 M_m     = lambda * Ker(ind_train,:)' + (1 / N) * Ker_x' * Ker_x;       % Ker_x * Ker_x' = Ker_x' * Ker_x - Hence either one works
 beta_m  = M_m \ b_m;
 
        
 for i = 1: 1 : N
     K(i)     = 0;
     for k = 1 : 1 : N_ker
         K(i)      = K(i)     + beta_m(k)  * Ker_x(i,k);      % Ker_x(pj,pi)
     end
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

