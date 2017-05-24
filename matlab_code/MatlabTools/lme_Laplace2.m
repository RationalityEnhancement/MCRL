function [lme,theta_MAP,H]=lme_Laplace2(log_joint,theta,H)
    %This function approximates the log model evidence ln p(y|m) by
    %applying the Laplace approximation to p(theta|y,m) and plugging it into
    %the equality p(y|m)=p(y,theta|m)/p(theta|y,m).
    
            
    options=optimoptions('fmincon','display','iter','algorithm','sqp');
    
    %[x,fval,exitflag,output,lambda,grad,hessian]: use hessian instead
    %of -H!
    [theta_MAP,fval,exitflag,output,lambda,grad,hessian]=fmincon(@(params) -log_joint(params),theta,[],[],[],[],[],[],[],options)
    
    if not(exist('H','var'))
        H = tapas_riddershessian(log_joint, theta_MAP);
    end
    %lme=log_joint(theta_MAP)+nr_parameters/2*log(2*pi)-1/2*logdet(-H)
    lme=log_joint(theta_MAP)+numel(theta_MAP)/2*log(2*pi)-1/2*logdet(-H);
    
    
end