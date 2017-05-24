function [x, fx, X_sample, F_sample, result] = IMGPO_stochastic_objective(f, d, x_input_domain, ...
          nb_iter, XI_max, ...
          GP_use, GP_kernel_est, GP_varsigma, likfunc, meanfunc, covfunc, hyp, ...
          result_diplay, result_save, GP_kernel_est_timing,...
          plot_func, plot_point, stop_per_iter)
% output: 
%      x = global optimizer 
%      fx = global optimal value f(x)
% optinal output:
%      X_sample = sampled points 
%      F_sample = sampled values of f 
%      result = intermidiate results 
%               for each iteration t, result(t,:) = [N, n (split #), fmax_hat, rho_bar, xi_max, depth_T, time(s)] 
% input:  
%      f = ovjective function (to be optimized)
%      d = input dimention of ovjective functio f
%      x_input_domain = input domain; 
%          e.g., = [-1 3; -3 3] means that domain(f) = {(x1,x2) : -1 <= x1 <= 3 and -3 <= x2 <= 3]}
%      nb_iter = the number of iterations to perform
% input: algorithm  
%      XI_max = to limit the computational time due to GP: 2^2 or 2^3 is computationally reasonable (see the NIPS paper for more detail)
% input: Gaussian process
%      GP_use = 1: use GP 
%      GP_kernel_est = 1: update kernel parameters  during execusion   
%      GP_varsigma: UCB = mean + GP_varsigma(M) * sigma
%                   e.g., nu = 0.05; GP_varsigma = @(M) sqrt(2*log(pi^2*M^2/(12*nu))); 
%                   e.g., GP_varsigma = @(M) 2; 
% input: gpml library (see the manual of gpml library for detail)
%      likfunc = likelihood function: e.g., @likGauss
%      meanfunc = mean function: e.g., @meanConst
%      covfunc = covariance function (kernel): e.g., {@covMaterniso, 5}
%      hyp = hyper-parameters: e.g., hyp.lik = -inf; hyp.mean = 0; hyp.cov = log([1/4; 1]);
% input display flag:
%      result_diplay = 1: print intermidiate results
%      result_save = 1: save intermidiate result and return as result
%      plot_func = 1: plot objective function if the dimensionality is <= 2
%      plot_point = 1; plot data points if the dimensionality is <= 2

h_max_f = @(N) inf;
%h_max_f = @(N) ceil(sqrt(N));

run gpml-matlab-v3.6-2015-07-07/startup 

addpath('supplementary');

%% initilisation
h_upper = 1000;

% variable initialization
clear x;
clear xxx;
clear xxx_d;
clear xxx_g;
clear X_sample;
clear F_sample;
clear UB;
clear LB;
clear C_sample;
clear t;
clear GPplot1;
clear GPplot2;

double t;
double X_sample;
double F_sample;
C_sample = [];
result = [];

N = 0;
M = 1;
split_n = 0;
z0= 1;

% scale factor caluculation
dumm = x_input_domain(:,2) - x_input_domain(:,1);
x_scale = [dumm x_input_domain(:,1) ]';

% plot function
if(plot_point || plot_func) 
    cla; 
    hold on
end
% plot function
if(d==1 && plot_func) draw_function(0,1,f,1,x_scale); end
if(d==2 && plot_func) draw_function2(0,1,f,x_scale); end



%% initilisation of the tree
t = cell(h_upper,1);
for i = 1:h_upper
    t{i}.x_max = [];
    t{i}.x_min = [];
    t{i}.x = [];
    
    t{i}.f = [];

    t{i}.leaf = [];
    t{i}.new = [];
    
    t{i}.node =[];
    t{i}.parent =[];
    
    t{i}.samp = [];
end
tic

    t{1}.x_min = zeros(1,d);
    t{1}.x_max = ones(1,d);
    t{1}.x = repmat(z0/2,1,d);

    for d_i=1:d
        xxx(1,d_i) = [t{1}.x(1,d_i) 1] * x_scale(:,d_i);  
    end

    fsample = f(xxx);
     
    t{1}.f = fsample;
    N = N + 1;
    t{1}.leaf = 1;
    split_n = split_n + 1;
    t{1}.samp = 1;
    
    X_sample = xxx;
    F_sample = fsample;
    LB = max(F_sample);
       
    if result_diplay == 1
      fprintf(1,'N = %d, n = %d, LB = %0.10f \n', N, split_n, LB); 
    end
    if  result_save == 1
        result = [result; N, split_n, LB, 0, 0, 0, toc];
    end
        
    if(d==1 && plot_point) plot(xxx, t{1}.f, 'k*','MarkerSize',3,'Color',[0 0 1]); end
    if(d==2 && plot_point) scatter3(xxx(1,2), xxx(1,1), t{1}.f,40,[0 0 1]); end

for h=1:h_upper  
  if size(t{h}.x,1) < 1, break, end
end
depth_T = h - 1;
%% execution
rho_avg = 0;
rho_bar = 0;
xi_max = 0;
time_t = 0;
LB_old = LB;
XI = 1;
while N < nb_iter
  i_max = zeros(depth_T,1);
  b_max = -inf * ones(depth_T,1);
  b_hi_max = -inf;
  time_t = time_t + 1;
  h_max = h_max_f(N); 
  %% steps (i)-(ii)
  for h=1:depth_T
    if h > h_max, break, end
    GP_label = 1;
    while GP_label == 1
      for i=1:size(t{h}.x,1) 
        if (t{h}.leaf(i) == 1)
          b_hi = t{h}.f(i);
          if (b_hi > b_hi_max)
            b_hi_max = b_hi;
            i_max(h) = i;
            b_max(h) = b_hi;          
          end        
        end   
      end
      if i_max(h) == 0, break, end
      if t{h}.samp(i_max(h)) == 1
        GP_label = 0; 
      else
        xxx = t{h}.x(i_max(h),:);
        for d_i=1:d
          xxx(1,d_i) = [xxx(1,d_i) 1] * x_scale(:,d_i);  
        end  
        fsample = f(xxx);
        t{h}.samp(i_max(h)) = 1;
        X_sample = [X_sample; xxx];
        F_sample = [F_sample; fsample];
        N = N+1;
        if d==1 && plot_point == 1 
          plot(xxx, fsample, 'k*','MarkerSize',10,'Color',[0 1 0]);
          plot(xxx, 0.01, 'k*','MarkerSize',10,'Color',[0 1 0]);
        end
        if d==2 && plot_point == 1 
          scatter3(xxx(1,2), xxx(1,1), fsample,20,[0 1 0],'filled');
        end
        if result_save == 1
          LB = max(F_sample);  
          result = [result; N, split_n, LB, rho_bar, xi_max, depth_T, toc];
        end
      end
    end
  end
  
  %% steps (iii)
  if GP_use == 1
    for h=1:depth_T        
     if h > h_max, break, end    
      if(i_max(h) ~= 0)
        % compute xi
        xi = 0;
        for h_2 = h + 1 : min(depth_T, h + min(ceil(XI),XI_max))
          if(i_max(h_2) ~= 0)
              xi = h_2 - h;
              break;
          end
        end
        % compute z_max = z(h,i^*_h)
        z_max = -inf;
        if xi ~=0 
          % prepare
          for h_2 = h : h + xi 
            t2{h_2}.x_max = [];
            t2{h_2}.x_min = [];
            t2{h_2}.x = [];  
          end
          t2{h}.x_max(1,:) = t{h}.x_max(i_max(h),:); 
          t2{h}.x_min(1,:) = t{h}.x_min(i_max(h),:); 
          t2{h}.x(1,:) = t{h}.x(i_max(h),:);
          
          % compute z_max by expanding GP tree
          M_2 = M; 
          for h_2 = h : h+xi-1 
            for i_2 = 1:3^(h_2-h)
              xx = t2{h_2}.x(i_2,:);  
              [~, splitd] = max(t2{h_2}.x_max(i_2,:) - t2{h_2}.x_min(i_2,:));
              x_g = xx;
              x_g(splitd) = (5 * t2{h_2}.x_min(i_2,splitd) + t2{h_2}.x_max(i_2,splitd))/6.0;
              x_d = xx;
              x_d(splitd) = (t2{h_2}.x_min(i_2,splitd) + 5 * t2{h_2}.x_max(i_2,splitd))/6.0;
              for d_i=1:d
                xxx_g(1,d_i) = [x_g(1,d_i) 1] * x_scale(:,d_i);  
                xxx_d(1,d_i) = [x_d(1,d_i) 1] * x_scale(:,d_i);  
              end
              [m_g, s2_g] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx_g);
              z_max = max(z_max, m_g+GP_varsigma(M_2)*sqrt(s2_g));
              M_2 = M_2 + 1;

              [m_d, s2_d] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx_d);
              z_max = max(z_max, m_d+GP_varsigma(M_2)*sqrt(s2_d));
              M_2 = M_2 + 1;

              if z_max >= b_max(h+xi), break, end
              
              t2{h_2+1}.x = [t2{h_2+1}.x;x_g];
              newmin = t2{h_2}.x_min(i_2,:);
              t2{h_2+1}.x_min = [t2{h_2+1}.x_min; newmin];
              newmax = t2{h_2}.x_max(i_2,:);
              newmax(splitd) = (2*t2{h_2}.x_min(i_2,splitd)+t2{h_2}.x_max(i_2,splitd))/3.0;
              t2{h_2+1}.x_max = [t2{h_2+1}.x_max; newmax];
                  
              t2{h_2+1}.x = [t2{h_2+1}.x;x_d];  
              newmax = t2{h_2}.x_max(i_2,:);
              t2{h_2+1}.x_max = [t2{h_2+1}.x_max; newmax];
              newmin = t2{h_2}.x_min(i_2,:);
              newmin(splitd) = (t2{h_2}.x_min(i_2,splitd)+2*t2{h_2}.x_max(i_2,splitd))/3.0;
              t2{h_2+1}.x_min = [t2{h_2+1}.x_min; newmin];
                  
              t2{h_2+1}.x = [t2{h_2+1}.x;xx];
              newmin = t2{h_2}.x_min(i_2,:);
              newmax = t2{h_2}.x_max(i_2,:);  
              newmin(splitd) = (2*t2{h_2}.x_min(i_2,splitd)+t2{h_2}.x_max(i_2,splitd))/3.0;
              newmax(splitd) = (t2{h_2}.x_min(i_2,splitd)+2*t2{h_2}.x_max(i_2,splitd))/3.0;
              t2{h_2+1}.x_min = [t2{h_2+1}.x_min; newmin];
              t2{h_2+1}.x_max= [t2{h_2+1}.x_max; newmax];
            end
            if z_max >= b_max(h+xi), break, end
          end
          
        end
        
        if xi ~= 0 && z_max < b_max(h+xi)  
          M = M_2;                           % if we actually used M_2 UCBs, we update M = M_2;
          i_max(h) = 0;                      % if it turns out that some UCB exceeded b_max(h+xi), then it does not matter whether or not f <= UCB. It may be UCB<f, and still it works exactly same. So, we do not have to update M in this case.   
          xi_max = max(xi,xi_max);
        end        
      end
    end
  end
  %% steps (iv)-(v)
  b_hi_max_2 = -inf;
  rho_t = 0;
  for h=1:depth_T   
   if h > h_max, break, end
   if(i_max(h) ~= 0 && b_max(h) > b_hi_max_2)
   %if(i_max(h) ~= 0)
    rho_t = rho_t + 1; 
    depth_T = max(depth_T,h+1);  
    split_n = split_n + 1;
  
    t{h}.leaf(i_max(h)) = 0;

    xx = t{h}.x(i_max(h),:);
  
    % --- find the dimension to split:  one with the largest range ---
   
    [~, splitd] = max(t{h}.x_max(i_max(h),:) - t{h}.x_min(i_max(h),:));
    x_g = xx;
    x_g(splitd) = (5 * t{h}.x_min(i_max(h),splitd) + t{h}.x_max(i_max(h),splitd))/6.0;
    x_d = xx;
    x_d(splitd) = (t{h}.x_min(i_max(h),splitd) + 5 * t{h}.x_max(i_max(h),splitd))/6.0;
  
    % --- splits the leaf of the tree ----
    % left node
    t{h+1}.x = [t{h+1}.x;x_g];
    for d_i=1:d
        xxx_g(1,d_i) = [x_g(1,d_i) 1] * x_scale(:,d_i);  
    end
    UCB = +inf;                                                             
    if GP_use == 1
      [m, s2] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx_g);
      UCB = m+(GP_varsigma(M)+0.2)*sqrt(s2);
    end
    if UCB <= LB && GP_use == 1
      M = M + 1;                         % need to update M only if we require f <= UCB. In the other case, f can be f > UCB and not require to take union bound. 
      fsample_g = UCB;
      t{h+1}.samp = [t{h+1}.samp 0];     
      samp_g = 0;
    else
      fsample_g = f(xxx_g);
      t{h+1}.samp = [t{h+1}.samp 1]; 
      samp_g = 1;
      
      X_sample = [X_sample; xxx_g];
      F_sample = [F_sample; fsample_g];
      N = N+1;
      b_hi_max_2 = max(b_hi_max_2, fsample_g);
      if  result_save == 1
        LB = max(F_sample);
        result = [result; N, split_n, LB, rho_bar, xi_max, depth_T, toc];
      end
    end
    t{h+1}.f = [t{h+1}.f fsample_g];
  
    newmin = t{h}.x_min(i_max(h),:);
    t{h+1}.x_min = [t{h+1}.x_min; newmin];
    newmax = t{h}.x_max(i_max(h),:);
    newmax(splitd) = (2*t{h}.x_min(i_max(h),splitd)+t{h}.x_max(i_max(h),splitd))/3.0;
    t{h+1}.x_max = [t{h+1}.x_max; newmax];
    t{h+1}.leaf = [t{h+1}.leaf 1];

    % right node
    t{h+1}.x = [t{h+1}.x;x_d];  
    for d_i=1:d
        xxx_d(1,d_i) = [x_d(1,d_i) 1] * x_scale(:,d_i);  
    end
    UCB = +inf;
    if GP_use == 1
      [m, s2] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx_d);
      UCB = m+(GP_varsigma(M)+0.2)*sqrt(s2);
    end
    if UCB <= LB && GP_use == 1
      M = M + 1;    
      fsample_d = UCB;
      t{h+1}.samp = [t{h+1}.samp 0]; 
      samp_d = 0;
    else
      fsample_d = f(xxx_d);
      t{h+1}.samp = [t{h+1}.samp 1];
      samp_d = 1;
      
      X_sample = [X_sample; xxx_d];
      F_sample = [F_sample; fsample_d];
      N = N+1;
      b_hi_max_2 = max(b_hi_max_2, fsample_d);
          
      if  result_save == 1
        LB = max(F_sample);  
        result = [result; N, split_n, LB, rho_bar, xi_max, depth_T, toc];
      end
    end
    t{h+1}.f = [t{h+1}.f fsample_d];
  
    newmax = t{h}.x_max(i_max(h),:);
    t{h+1}.x_max = [t{h+1}.x_max; newmax];
    newmin = t{h}.x_min(i_max(h),:);
    newmin(splitd) = (t{h}.x_min(i_max(h),splitd)+2*t{h}.x_max(i_max(h),splitd))/3.0;
    t{h+1}.x_min = [t{h+1}.x_min; newmin];
    t{h+1}.leaf = [t{h+1}.leaf 1];
   
    % central node 
    t{h+1}.x = [t{h+1}.x;xx];
    t{h+1}.f = [t{h+1}.f t{h}.f(i_max(h))];
    t{h+1}.samp = [t{h+1}.samp 1];
    newmin = t{h}.x_min(i_max(h),:);
    newmax = t{h}.x_max(i_max(h),:);  
    newmin(splitd) = (2*t{h}.x_min(i_max(h),splitd)+t{h}.x_max(i_max(h),splitd))/3.0;
    newmax(splitd) = (t{h}.x_min(i_max(h),splitd)+2*t{h}.x_max(i_max(h),splitd))/3.0;
    t{h+1}.x_min = [t{h+1}.x_min; newmin];
    t{h+1}.x_max= [t{h+1}.x_max; newmax];
    t{h+1}.leaf = [t{h+1}.leaf 1];
    
  %% plot new sample points
    %% 1D plot
    if(d==1 && plot_point)
      if samp_g == 1
        plot(xxx_g, fsample_g, 'k*','MarkerSize',8,'Color',[0 0 1]);
        plot(xxx_g, 0.01, 'k*','MarkerSize',8,'Color',[0 0 1]);
      else
        plot(xxx_g, fsample_g, 'k*','MarkerSize',8,'Color',[1 0 0]);
        plot(xxx_g, 0.01, 'k*','MarkerSize',8,'Color',[1 0 0]);
      end
      if samp_d == 1
        plot(xxx_d, fsample_d, 'k*','MarkerSize',8,'Color',[0 0 1]);
        plot(xxx_d, 0.01, 'k*','MarkerSize',8,'Color',[0 0 1]);
      else
        plot(xxx_d, fsample_d, 'k*','MarkerSize',8,'Color',[1 0 0]);
        plot(xxx_d, 0.01, 'k*','MarkerSize',8,'Color',[1 0 0]);
      end
      hold on
    
    % GP plot
      if exist('GPplot2')
        delete(GPplot2)             
      end
    
      z = linspace(x_input_domain(1), x_input_domain(2), 1001)';
      [m s2] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, z);
      %GPCI = [m+GP_varsigma*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
      %GPplot1 = fill([z; flipdim(z,1)], GPCI, [7 7 7]/8);
      %uistack(GPplot1,'bottom');
      %hold on
      %GPplot2 = plot(z, m, 'LineWidth', 1, 'Color', [1 0 0]); 
    
      GPplot2 = plot(z, m+GP_varsigma(M)*sqrt(s2), 'LineWidth', 1, 'Color', [1 0 0]); 
    
      axis off
    
    end
    %% 2D plot
    if(d==2 && plot_point)
      if samp_g == 1  
        scatter3(xxx_g(1,2), xxx_g(1,1), fsample_g,40,[0 0 1]);
      else 
        scatter3(xxx_g(1,2), xxx_g(1,1), fsample_g,40,[1 0 0]);
      end
      if samp_d == 1
        scatter3(xxx_d(1,2), xxx_d(1,1), fsample_d,40,[0 0 1]);
      else
        scatter3(xxx_d(1,2), xxx_d(1,1), fsample_d,40,[1 0 0]);
      end
    end
  
    
    if exist('GPplot99')
       delete(GPplot99);
    end
    %f_gp = @(x) gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, x);
    %GPplot99 = draw_function2_GP(0,1,f_gp,x_scale,GP_varsigma(M)); 
    %draw_function2_GP(0.35,0.45,f_gp,x_scale,GP_varsigma(M));
  
    %% --- output results -------------------------------------------------------
    LB = max(F_sample);
    
    if result_diplay == 1
      fprintf(1,'%d, N = %d, n = %d, fmax_hat = %0.10f, rho = %d, xi = %d,h = %d, time = %f \n', time_t, N, split_n, LB, rho_bar, xi_max, depth_T, toc);       
    end
    
    if stop_per_iter == 1, pause; end
    %-------------------------------------------------------------------------- 
      
    end
  end
  rho_avg = (rho_avg * (time_t - 1) + rho_t) / time_t;
  rho_bar = max(rho_bar,rho_avg);
  
  %% update Xi
  if LB_old == LB
      XI = max(XI - 2^-1,1);  
  else
      XI = XI + 2^2;
  end
  LB_old = LB;
  
  %% update GP hypper parameters
  if GP_kernel_est == 1
    warning ('off','all'); 
    if max(split_n == GP_kernel_est_timing)
        hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, [], X_sample, F_sample);
    end
    warning ('on','all');  
  end
  
  
end

%% get a maximum point
f_hi_max = -inf;
for h=1:h_upper  
  if size(t{h}.x,1) < 1, break, end
  for i=1:size(t{h}.x,1) 
    f_hi = t{h}.f(i);  
    if (f_hi > f_hi_max)
      f_hi_max = f_hi;
      i_max = i;
      h_max = h;
    end          
  end
end
h=h-1;
fprintf(1,'constructed a tree with depth h = %f \n', h);

x = t{h_max}.x(i_max,:);
for d_i=1:d
  xxx(1,d_i) = [x(1,d_i) 1] * x_scale(:,d_i);  
end
x = xxx;
fx = f_hi_max;

for i=1:size(X_sample,1)
  [E_f(i), s(i)] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, X_sample(i,:));
end

[max_val,max_pos]=max(E_f);
x=X_sample(max_pos,:);
fx=max_val;