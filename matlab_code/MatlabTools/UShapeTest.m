function [minimum_point,ci]=UShapeTest(y,x,alpha,covariates)
%Test whether y is a U-shaped function of x according to Lind and Mehlum
%(2009).
%CI of ratio based on Bebu, Seillier-Moiseiwitsch, and Thomas Mathew (2009)
%t_beta,t_gamma,p_beta,p_gamma

%construct regressors
N=numel(y);
X=zeros(N,3+size(covariates,2));
X(:,1)=x(:);
X(:,2)=x(:).^2;
X(:,3)=ones(N,1);
X(:,4:end)=covariates;
k=size(X,2);

%regress y on x and x^2
[beta,~,E] = regress(y(:),X);

%estimate the minimum point of the regression function

minimum_point=-1/2*beta(1)/beta(2);

%determine confidence interval of the minimum point
t=tinv(alpha/2,N-k);

sigma2=sum(E.^2)/(N-k);
A=inv(X'*X);
beta1=beta(1);
beta2=beta(2);

a=beta2^2-sigma2*A(2,2)*t^2;
b=-2*beta1*beta2+2*t^2*sigma2*A(1,2);
c=beta1^2-t^2*sigma2*A(1,1);


root1=(-b+sqrt(b^2-4*a*c))/(2*a);
root2=(-b-sqrt(b^2-4*a*c))/(2*a);

ci=[min(-1/2*root1,-1/2*root2),max(-1/2*root1,-1/2*root2)];

end