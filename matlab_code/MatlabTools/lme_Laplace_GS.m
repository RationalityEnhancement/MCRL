function [lme,theta_MAP,H]=lme_Laplace_GS(log_joint,nr_parameters,boundary_low,boundary_high,x_init)
    %This function approximates the log model evidence ln p(y|m) by
    %applying the Laplace approximation to p(theta|y,m) and plugging it into
    %the equality p(y|m)=p(y,theta|m)/p(theta|y,m).
    
    gs=GlobalSearch('MaxTime',14395,'TolFun',1e-1);
    opts = optimoptions('fmincon','Algorithm','sqp');
    
    if not(exist('boundary_low','var'))
        boundary_low=-realmax*ones(nr_parameters,1);
    end
    if not(exist('boundary_high','var'))
        boundary_high=realmax*ones(nr_parameters,1);
    end
    
    if and(exist('boundary_low','var'),exist('boundary_high','var'))
        x0=(boundary_low+boundary_high)/2+(rand(nr_parameters,1)-0.5).*(boundary_low+boundary_high)/2;
    else
        x0=ones(nr_parameters,1);
    end
    
    if exist('x_init','var')
        x0=x_init;
    end
        
    options=optimoptions('fmincon','display','iter','algorithm','sqp','TolFun',1e-1);
    if nr_parameters==1
        %[x,fval,exitflag,output,lambda,grad,hessian]: use hessian instead
        %of -H!
            problem = createOptimProblem('fmincon','objective', ...
    @(params) -log_joint(params(1)),'x0',x0,'lb',boundary_low,'ub',boundary_high, ...
    'options',options);

        [theta_MAP,~,~,~,solsgs] = run(gs,problem);
        [theta_MAP,fval,exitflag,output,lambda,grad,hessian]=fmincon(@(params) -log_joint(params(1)),theta_MAP,[],[],[],[],boundary_low,boundary_high,[],options)
        %H = tapas_riddershessian(@(params) log_joint(params(1)), theta_MAP)
        %lme=log_joint(theta_MAP)+nr_parameters/2*log(2*pi)-1/2*logdet(-H)
        lme=log_joint(theta_MAP)+nr_parameters/2*log(2*pi)-1/2*logdet(hessian);
    elseif nr_parameters==2
        %theta_MAP=fmincon(@(params) -log_joint(params(1),params(2)),x0,[],[],[],[],boundary_low,boundary_high,[],options)
        problem = createOptimProblem('fmincon','objective', ...
    @(params) -log_joint(params(1),params(2)),'x0',x0,'lb',boundary_low,'ub',boundary_high, ...
    'options',options);
        [theta_MAP,~,~,~,solsgs] = run(gs,problem);
        [theta_MAP,fval,exitflag,output,lambda,grad,hessian]=...
            fmincon(@(params) -log_joint(params(1),params(2)),theta_MAP,[],[],[],[],boundary_low,boundary_high,[],options);
        %H = tapas_riddershessian(@(params) log_joint(params(1),params(2)), theta_MAP);
        %lme=log_joint(theta_MAP(1),theta_MAP(2))+nr_parameters/2*log(2*pi)-1/2*logdet(-H)
        lme=log_joint(theta_MAP(1),theta_MAP(2))+nr_parameters/2*log(2*pi)-1/2*logdet(hessian)
    elseif nr_parameters==3
        problem = createOptimProblem('fmincon','objective', ...
    @(params) -log_joint(params(1),params(2),params(3)),'x0',x0,'lb',boundary_low,'ub',boundary_high, ...
    'options',options);
        [theta_MAP,~,~,~,solsgs] = run(gs,problem);
        
        [theta_MAP,fval,exitflag,output,lambda,grad,hessian]...
            =fmincon(@(params) -log_joint(params(1),params(2),params(3)),theta_MAP,[],[],[],[],boundary_low,boundary_high,[],options)
        %theta_MAP=fmincon(@(params) -log_joint(params(1),params(2),params(3)),x0,[],[],[],[],boundary_low,boundary_high,[],options)
        %H = tapas_riddershessian(@(params) log_joint(params(1),params(2),params(3)), theta_MAP)
        %lme=log_joint(theta_MAP(1),theta_MAP(2),theta_MAP(3))+nr_parameters/2*log(2*pi)-1/2*logdet(-H)
        lme=log_joint(theta_MAP(1),theta_MAP(2),theta_MAP(3))+nr_parameters/2*log(2*pi)-1/2*logdet(hessian)
    end
    
    H=-hessian;
    
end