function VPI=valueOfPerfectInformationBernoulli(alpha,beta)
%alpha: alpha parameter of the Bernoulli distributions over the
%probability of the light being on
%beta: beta parameter of the Bernoulli distribtion over the probability of the light being on
%c: index of the action about which perfect information is being obtained

E_correct=[alpha,beta]/(alpha+beta);
[max_val,max_pos]=max(E_correct);

EV_original_preference = @(x) (alpha>beta)*x+(beta>alpha)*(1-x)+(alpha==beta)*0.5;


VPI = integral(@(x) betapdf(x,alpha,beta).*(max(x,1-x)-EV_original_preference(x)),0,1);    

end