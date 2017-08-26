function VPI=valueOfPerfectInformationBernoulli(alpha,beta,r_correct,r_wrong)
%alpha: alpha parameter of the Bernoulli distributions over the
%probability of the light being on
%beta: beta parameter of the Bernoulli distribtion over the probability of the light being on
%r_correct: reward for making an correct prediction
%r_wrong: reward for making an incorrect prediction

if not(exist('r_correct','var'))
    r_correct=1;
end

if not(exist('r_wrong','var'))
    r_wrong=-1;
end

%E_correct=[alpha,beta]/(alpha+beta);
%[max_val,max_pos]=max(E_correct);

EV_original_preference =...
    @(p) (alpha>beta)*(p*r_correct+(1-p)*r_wrong)+... %if on seems more likely than off
    (beta>alpha)*((1-p)*r_correct+p*r_wrong)+... %if off seems more likely than on
    (alpha==beta)*(0.5*r_correct+0.5*r_wrong); %if both seem equally likely


VPI = integral(@(x) betapdf(x,alpha,beta).*... %probability of p
    ((max(x,1-x)*r_correct+min(x,1-x)*r_wrong)-... %expected reward if p was known
    EV_original_preference(x)),0,1);  %expected reward in the current knowledge state  

end