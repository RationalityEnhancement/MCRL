function [results]=strategyDetection(data,model_predictions)
%Infer each participant's propensity to use each of the strategies.
%model_predictions: possible responses x trials x models
%data: struct with field choices, struct.choices: nr_subjects x nr_trials

addpath('/Users/Falk/Dropbox/PhD/MatlabTools/spm8/')

nr_models=size(model_predictions,3);
for s=1:data.nr_subjects
    for m=1:nr_models
        
        
        for t=1:data.nr_trials
            
            data.likelihood(t,s,m)=model_predictions(data.choices(s,t),t,m);
        end
        
        data.log_p_y_given_m(:,1,s)=log(min(1-eps,max(eps,data.likelihood(:,s,m))));
    end
    
    
    [alpha(:,s),exp_r(:,s),xp(:,s)]=spm_BMS(data.log_p_y_given_m(:,:,s),1e7,true);
    
    for m=1:nr_models
        CI_proportion(:,m,s)=betaHPDInterval([alpha(m,s),sum(alpha(:,s))-alpha(m,s)],0.95);
        proportion_error_bars(m,1,:,s)=abs(CI_proportion(:,m,s)-exp_r(m,s));
    end
end


results.exp_r=exp_r;
results.xp=xp;
results.CI_proportion=CI_proportion;
results.proportion_error_bars=proportion_error_bars;

end