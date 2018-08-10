function [eta K] = gain_rkhs_zm_mcmc( Xi , c , d, kernel, lambda, epsilon, N_ker, diag)
% Returns the gain computed at particle locations Xi using an RKHS 
tic;
N = length(Xi);

% Evaluation of kernel matrices 
if kernel == 0  % Gaussian
    for i = 1:N
       for k = 1:N_ker          
           if (Xi(i) ~= Xi(k)) || (i == k)
               Ker(i,k) =  exp(-(norm(Xi(i) - Xi(k)).^2/(4 * epsilon)));    
               for d_i = 1 : d
                   Ker_x(i,k,d_i) =  - (Xi(i,d_i) - Xi(k,d_i)) / (2 * epsilon) .* Ker(i,k);
               end
           end
       end
    end 
end  
   
H     = c(Xi);       
eta   = mean(c(Xi));
Y     =  (H - eta)';
K_hat = zeros(1,d);
for i = 1:N
    K_hat = K_hat + (1/N) * ((c(Xi(i,:)) - eta) .* Xi(i,:));      % Constant gain approximation
end

% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
Ker_x_sum = zeros(N_ker,N_ker);
for d_i = 1 : d
    Ker_x_ones(:,d_i) =  Ker_x(:,:,d_i) * ones(N_ker,1);
    Ker_x_sum         =  Ker_x_sum + Ker_x(:,:,d_i)'* Ker_x(:,:,d_i);
    M(1 : N_ker, N_ker + d_i) =  Ker_x_ones(1 : N_ker, d_i);
    M(N_ker + d_i, 1 : N_ker) =  Ker_x_ones(1 : N_ker, d_i)';
end
b(1 : N_ker, :)            =  (2/N) * Ker' * Y'- (2/N) * Ker_x_ones (1:N_ker) * K_hat';
b(N_ker+1 : N_ker+d, :)    =  zeros(d,1);    
M(1 : N_ker, 1 : N_ker)    =  2 * lambda * Ker (1:N_ker, :) + ( 2 / N) * Ker_x_sum;       % Ker_x * Ker_x' = Ker_x' * Ker_x - Hence either one works
beta               =  M \ b;
        
 for i = 1: 1 : N
     K(i,:)  = K_hat;
     for k = 1 : 1 : N_ker
          for d_i = 1 : d
             K(i,d_i)      = K(i,d_i)     + beta(k)  * Ker_x(i,k,d_i);      % Ker_x(pj,pi)
         end           
     end
 end

toc
 
%% For displaying figures
if diag == 1
    figure;
    plot(Xi,K,'b*');    
end
  
end


