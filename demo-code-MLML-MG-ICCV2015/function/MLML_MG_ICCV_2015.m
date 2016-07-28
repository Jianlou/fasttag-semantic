function [Z_final, Z_cell, obj_L, obj_Z] = MLML_MG_ICCV_2015(Y,L_x, L_c, lambda_x, lambda_c, Phi_matrix, options)


 [num_c, num_sample] = size(Y);

Y_bar = 2.*Y-1; % transform to [-1, 1]

global positive_label_weight 
weight_matrix = ones(size(Y)); 
weight_matrix(Y==1) = positive_label_weight; 
Yw = double(weight_matrix .* Y_bar ); 

const = 2 * trace(Yw * Y'); 

A = -2 .* Yw; % m x n
B = lambda_x .* L_x; % n x n
C = lambda_c .* L_c; % m x m
D = Phi_matrix; % m x r

%% initialization of Z0
given_locations = (Y~=0.5);
if isfield(options, 'initialization')
    switch options.initialization
        case  'random'
            Z0 = rand(num_c, num_sample);
            
        case 'zero'
            Z0 = zeros(num_c, num_sample); 
    end
    Z0(given_locations) = Y(given_locations); 
else
    Z0 = Y; Z0(Z0 == 0.5) =0; 
end
Z0 = sparse(Z0);
 

if D == 0 % No semantic hierarchy constraint
        % call the MLML-PGD method
        options_pgd = options;
        [ Z_final, Z_cell, obj_Z ] = mlml_pgd_template(A, B , C, Z0, options_pgd); 
        obj_Z = obj_Z + const;
        obj_L = obj_Z;
else % With semantic hierarchy constraint
      % call MLML-ADMM
        options_admm = options;
        tic
        [Z_final, Z_cell, obj_L, obj_Z] = SolveQuadProg(Z0, A,B, C,D, options_admm);
        toc
        obj_L = obj_L + const; 
        obj_Z = obj_Z + const;
end


function [Z_final, Z_cell, obj_L, obj_Z] = SolveQuadProg(Z0, A,B,C,D, options)

[m,n]=size(Z0);
r = size(D, 2); % the number of edges
rho = options.rho_0;

Z_old = Z0;  
Lambda_matrix_old = zeros(r,n);
%Q_old = zeros(r,n);
Q_old = max(0, D'*Z_old);

D_D = D * D';
A_bar = A + D * ( Lambda_matrix_old - rho .*  Q_old ); 
B_bar = B; 
C_bar = C + 0.5 * rho .* D_D; 

t = 1;
const = trace( Q_old * (Lambda_matrix_old + 0.5 * rho .* Q_old)' );
obj_L(1) = obj_compute(A_bar, B_bar, C_bar, Z_old) + const; 
obj_Z(1) = obj_compute(A, B, C, Z_old);
fprintf('iter:%d, obj_L:%.2f, obj_Z:%.2f\n',0,obj_L(1), obj_Z(1));

Z_cell{1} = Z_old;

max_iter = options.max_iter_overall;
for iter =1:max_iter
     
    %% step 1, update Z_{k+1}
    A_bar = A + D * ( Lambda_matrix_old - rho .*  Q_old ); 
    C_bar = C + 0.5 * rho .* D_D; 

    [ Z_new, ~, obj_Lz ] = mlml_pgd_template(A_bar, B_bar , C_bar , Z_old, options); 

    %% update Q
    Q_new = max(0, D'*Z_new + Lambda_matrix_old ./ rho);
    
    %% update Lambda_matrix
     Lambda_matrix_new = Lambda_matrix_old + rho .* (D' * Z_new - Q_new);
    
    %% compute the objective function of L(Z, Q, Lambda_matrix)
    if ((iter-1)/5) == fix((iter-1)/5)
         A_bar = A + D * ( Lambda_matrix_new - rho .*  Q_new ); 
    
        const = trace( Q_new * (Lambda_matrix_new + 0.5 * rho .* Q_new)' );
        obj_L(t+1) = obj_compute(A_bar, B_bar, C_bar, Z_new) + const; 
        obj_Z(t+1) = obj_compute(A, B, C, Z_new);
        Z_cell{t+1} = Z_new;
        
        obj_diff = abs( (obj_L(t+1) - obj_Z(t+1))/obj_Z(t+1)); 
        T = D' * Z_new - Q_new; 
        dist = norm(T,'fro');
        fprintf('iter:%d, dist:%f, rho:%.2f, obj_L:%.2f, obj_Z:%.2f\n',iter,dist,rho, obj_L(t+1), obj_Z(t+1));
        
        if( iter>50 && dist<1e-8 ) %obj_diff<1e-15
            break; 
        else
            t = t+1;
        end         
    end
        
    Z_old = Z_new; 
    Q_old = Q_new; 
    Lambda_matrix_old = Lambda_matrix_new; 

    if(~mod(iter,options.rho_gap))
        rho = min(1e8,rho * options.rho_rate);
    end 

end
Z_final = Z_new;


function [Z_final, Z_cell, obj, alpha] = mlml_pgd_template(A, B, C, Z0, options)

% min_Z { tr(A' * Z) + tr(Z B Z') + tr(Z' C Z) }, st. 0 <= Z <= 1

max_iter = options.max_iter;
gap_compute = options.gap_compute;
rate_step = options.rate_step;
alpha_rate = options.alpha_rate;

obj = zeros(1, max_iter);
obj(1) = obj_compute(A, B, C, Z0); 

alpha = zeros(1, max_iter);
%alpha_hessian_eig = 1/max(eigs(B,1), eigs(C,1)); 

Z_old = Z0;
Z_cell{1} = Z_old;

for i=1:max_iter
    
    % step 1, gradient
    Z_gradient = A + 2.* Z_old * B + 2.* C * Z_old; 
    
    % step 2, step size by exact line search
    if ((i-1)/gap_compute) == fix((i-1)/gap_compute)
          alpha(i) = ( step_size_compute(A, B, C, Z_old, Z_gradient) ) / (alpha_rate*i); %*options.iter
    else
        alpha(i) = alpha(i-1) * rate_step;
    end
    
    % step 3, projection to [0,1] 
    Z_new = Z_old - alpha(i) .* Z_gradient;
    
    % step 4, projection to [0,1] 
    Z_new = min(1,  max(0, Z_new) );
    Z_cell{i+1} = Z_new;
    
    % step 5, objective function and check convergence
    if ((i-1)/1) == fix((i-1)/1)
        obj(i+1) = obj_compute(A, B, C, Z_new); 
        
        obj_diff=abs(obj(i+1)-obj(i+1 - 1))/abs(obj(i));
         if (obj_diff<1*10^(-8))
           break;
        else
           Z_old = Z_new;
        end       
    end
        
end

Z_final=Z_new;
obj( obj == 0) = [];

% alpha( alpha == 0) =[]; 
% figure; plot(1:length(alpha), alpha, '-go'),
% xlabel('Iteration'), ylabel('\alpha'),
% title('\alpha');

function obj = obj_compute(A, B, C, Z)

obj = trace(Z * A') + trace( (Z * B) * Z') + trace( C * (Z * Z') );

function alpha = step_size_compute(A, B, C, Z, Z_gradient)

% the step size by exact line search for   tr(A' * Z) + tr(Z B Z') + tr(Z' C Z)

numerator = 0.5 * trace( Z_gradient * A') + trace( (Z * B) * Z_gradient' ) + trace( C * (Z_gradient * Z') );
dom = trace( (Z_gradient * B) * Z_gradient' ) + trace( C * (Z_gradient * Z_gradient') );

alpha = numerator / dom; 
