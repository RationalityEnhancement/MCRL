function lme=lmeLaplace(log_joint,gradient,Hessian)

theta0=randn(log_joint.nr_parameters,1);

if exist('gradient','var')
    
    log_joint_and_grad=@(theta) [log_joint.density(theta);gradient(theta)];
    
    [theta_MAP,fval,exitflag,output,grad,hessian]=fminunc(log_joint_and_grad,theta0,optimset('GradObj','on'));
else
    [theta_MAP,fval,exitflag,output,grad,hessian]=fminunc(log_joint.density,theta0);
end

    lme=power(2*pi,log_joint.nr_parameters/2).*power(det(Hessian),1/2).*...
        log_joint.density(theta_MAP);

end