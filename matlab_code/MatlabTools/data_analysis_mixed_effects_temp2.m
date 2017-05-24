model_nr=7+whichinf;
%analyse Jane's data

addpath('/cluster/home/phys/flieder/MatlabTools/')

%each row corresponds to one trial
%first column: average rating of the first restaurant
%second column: #ratings of the first restaurant
%third column: average rating of the second restaurant
%fourth column: #ratings of the second restaurant

options=csvread('/cluster/home/phys/flieder/PreferenceLearning/options_Matlab_friendly.csv');
nr_trials=size(options,1);
%each row corresponds to one trial
%each column correspondsto one subject
decisions=csvread('/cluster/home/phys/flieder/PreferenceLearning/decisions_Matlab_friendly.csv')';
chose_first_option=(decisions==1);
nr_subjects=size(decisions,2);

%% define models
sigma_prior=10;

models(1).X=repmat(options(:,1)-options(:,3),[nr_subjects,1]);
models(2).X=repmat(options(:,2)-options(:,4),[nr_subjects,1]);
models(3).X=repmat([options(:,1)-options(:,3),options(:,2)-options(:,4)],[nr_subjects,1]);
models(4).X=repmat([options(:,1).*options(:,2)-options(:,3).*options(:,4)],[nr_subjects,1]);
models(5).X=repmat([options(:,1)-options(:,3),options(:,2)-options(:,4),options(:,1).*options(:,2)-options(:,3).*options(:,4)],[nr_subjects,1]);

mu_prior=3.62; tau_prior=2.61;
tau_likelihood=0.65;
[mu_Delta,tau_Delta]=NormalNormalModel.posteriorBeliefAboutDifference(mu_prior,tau_prior,tau_likelihood,options(:,1),options(:,2),options(:,3),options(:,4));
models(6).X=repmat(sqrt(tau_Delta).*mu_Delta,[nr_subjects,1]);
models(7).X=repmat([sqrt(tau_Delta).*mu_Delta,options(:,2)-options(:,4)],[nr_subjects,1]);

model8=GeneralizedProbitModel(8,sigma_prior);
model9=GeneralizedProbitModel(9,sigma_prior);

for m=1:9
    models(m).sigma_prior=sigma_prior;
end

%% compute model evidences
nr_samples=1e6; %for the final results
parfor s=1:nr_subjects

    first_trial=(s-1)*nr_trials+1;
    last_trial=s*nr_trials;
    
    if model_nr<8
        [BIC(s),model_evidence_approx_BIC(s),model_fit(s)]=ProbitModel.BIC(chose_first_option(first_trial:last_trial)',models(model_nr).X(first_trial:last_trial,:));    
        [lme(s),posterior_mean{s},posterior_var{s},~]=ProbitModel.approximateLogModelEvidence(chose_first_option(first_trial:last_trial)',models(model_nr).X(first_trial:last_trial,:),models(model_nr).sigma_prior,nr_samples);
    elseif model_nr==8
        [BIC(s),model_evidence_approx_BIC(s),model_fit(s)]=model8.BIC(chose_first_option(first_trial:last_trial)'); 
        [lme(s),posterior_mean{s},posterior_var{s},~]=model8.approximateLogModelEvidence(chose_first_option(first_trial:last_trial)',nr_samples);
    elseif model_nr==9
        [BIC(s),model_evidence_approx_BIC(s),model_fit(s)]=model9.BIC(chose_first_option(first_trial:last_trial)'); 
        [lme(s),posterior_mean{s},posterior_var{s},~]=model9.approximateLogModelEvidence(chose_first_option(first_trial:last_trial)',nr_samples);        
    end
end

save(['log_model_evidences_model',int2str(model_nr)],'lme')
save(['model_evidence_approx_BIC_model',int2str(model_nr)],'model_evidence_approx_BIC')
save(['model_fit_model',int2str(model_nr)],'model_fit')

parameter_estimates.posterior_mean=posterior_mean;
parameter_estimates.posterior_var=posterior_var;
save(['parameter_estimates_model',int2str(model_nr)],'parameter_estimates')