function [p_X_greater_Y,p_Y_greater_X,E_X,E_Y]=betaTest2(X,Y)

alpha0=[1,1];
x_values=unique(X);
y_values=unique(Y);

if length(x_values)<2
    x_values=unique([x_values,[0,1]]);
end
if length(y_values)<2
    y_values=unique([y_values,[0,1]]);
end

alpha_X=alpha0+[sum(X==x_values(2)),sum(X==x_values(1))];
alpha_Y=alpha0+[sum(Y==y_values(2)),sum(Y==y_values(1))];

E_X=alpha_X(1)/(alpha_X(1)+alpha_X(2));
E_Y=alpha_Y(1)/(alpha_Y(1)+alpha_Y(2));

p_X_greater_Y=integral(@(p) betapdf(p,alpha_X(1),alpha_X(2)).*betacdf(p,alpha_Y(1),alpha_Y(2)),0,1);
p_Y_greater_X=integral(@(p) betapdf(p,alpha_Y(1),alpha_Y(2)).*betacdf(p,alpha_X(1),alpha_X(2)),0,1);

end