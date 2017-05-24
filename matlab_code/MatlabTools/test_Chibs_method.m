%test the method by Chib and Jeliazkov

%define different models
max_nr_regressors=3;
X=rand(100,max_nr_regressors);
beta=randn(1,max_nr_regressors);

%model i uses the first i regressors
nr_models=max_nr_regressors;
models=[true false false; true true false; true true true];

%generate data
for m=1:nr_models
    data(:,m)=ProbitModel.generateData(X(:,models(m,:)),beta(models(m,:)));
end

%{
prior and likelihood are structs with the following fields:
  -density: function handle returning probability density
  -log_density: function handle returning log probability density
%}
for i=1:nr_models
    prior(i).density=@(betas) prod(normpdf(betas'),1);
    prior(i).log_density= @(betas) sum(log(max(eps,normpdf(betas'))),1);
    likelihood(i).density=@(y,betas) normcdf(X(y==1,:)*betas')*(1-normcdf(X(y==0,models(i,models(i,:)))*betas'));
    likelihood(i).log_density= @(y,betas) sum(log(max(eps,normcdf(X(y==1,models(i,:))*betas'))))+sum(log(1-min(1-eps,normcdf(X(y==0,models(i,:))*betas'))));
    likelihood(i).nr_parameters=sum(models(i,:));
end


%% estimate model evidences
%set up inversion method
%{
proposal is a struct with the following fields:
  -density= @(from,to,params)
  -log_density= @(from,to,params)
  -propose= @(from,params)
  -params
%}
proposal.density=@(from,to,sigma) mvnpdf((to-from)/sigma);
proposal.log_density=@(from,to,sigma) sum(log(max(eps,normpdf((repmat(to,[max(size(to,1),size(from,1))/size(to,1),1])-repmat(from,[max(size(to,1),size(from,1))/size(from,1),1]))/sigma)))',1);
proposal.propose=@(from,sigma) mvnrnd(from,sigma*diag(ones(length(from),1)));
proposal.sample=@(from,sigma,nr_samples) mvnrnd(from,sigma*diag(ones(length(from),1)),nr_samples);
proposal.params=1;
nr_samples=101000; burn_in=1000;
sigma_p=1; %prior STD
for true_model=1:nr_models
    for hypothesis=1:nr_models
        [lme(true_model,hypothesis),samples{true_model,hypothesis}]=...
            ProbitModel.approximateLogModelEvidence(data(:,true_model),X(:,models(hypothesis,:)),sigma_p,nr_samples)
        %[lme(true_model,hypothesis),SE(true_model,hypothesis),samples_posterior{true_model,hypothesis}]=...
        %    lmeChibJeliazkov01(data(:,true_model),prior(hypothesis),likelihood(hypothesis),proposal,nr_samples)
    end
end

imagesc(lme)

%% compute BIC scores
for true_model=1:nr_models
    for hypothesis=1:nr_models
        [BIC(true_model,hypothesis),model_evidence_approx_BIC(true_model,hypothesis)]=...
            ProbitModel.BIC(data(:,true_model),X(:,models(hypothesis,:)));
    end
end

figure(),imagesc(model_evidence_approx_BIC)