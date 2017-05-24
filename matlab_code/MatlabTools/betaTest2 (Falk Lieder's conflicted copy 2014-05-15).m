function [p_same,p_X_greater_Y,p_Y_greater_X,E_X,E_Y]=betaTest2(X,Y)

alpha0=[1,1];
x_values=unique(X);
y_values=unique(Y);
alpha_X=alpha0+[sum(X==x_values(1)),sum(X==x_values(2))];
alpha_Y=alpha0+[sum(Y==y_values(1)),sum(Y==y_values(2))];

E_X=alpha_X(1)/(alpha_X(1)+alpha_X(2));
E_Y=alpha_Y(1)/(alpha_Y(1)+alpha_Y(2));

dp=0.001;
ps=0:dp:1;

p_same=dp*dot(betapdf(ps,alpha_X(1),alpha_X(2)),betapdf(ps,alpha_Y(1),alpha_Y(2)));
p_X_greater_Y=dp*dot(betapdf(ps,alpha_X(1),alpha_X(2)),betacdf(ps,alpha_Y(1),alpha_Y(2)));
p_Y_greater_X=dp*dot(betapdf(ps,alpha_Y(1),alpha_Y(2)),betacdf(ps,alpha_X(1),alpha_X(2)));

end