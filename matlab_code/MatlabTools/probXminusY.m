function p_difference=probXminusY(P_x,P_y)
%This function computes the probability mass function of X-Y from the
%probability mass functions of X and Y.

dx=P_y.values(2)-P_y.values(1);
P_minus_y.values=-max(P_y.values):(-min(P_y.values));
P_minus_y.pmf=P_y.pmf(end:-1:1);

p_difference.values=(min(P_x.values)+min(P_minus_y.values)):dx:(max(P_x.values)+max(P_minus_y.values));
p_difference.pmf=conv(P_x.pmf(:),P_minus_y.pmf(:));

end