function [p_greater_half, p_smaller_half,E_pi]=betaTest(X)

alpha0=[1,1];
x_values=unique(X);
alpha_X=alpha0+[sum(X==x_values(1)),sum(X==x_values(2))];

E_pi=alpha_X(1)/(alpha_X(1)+alpha_X(2));

p_greater_half=1-betacdf(0.5,alpha_X(1),alpha_X(2));
p_smaller_half=betacdf(0.5,alpha_X(1),alpha_X(2));

end