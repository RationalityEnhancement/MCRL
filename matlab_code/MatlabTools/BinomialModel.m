%Extract the paradigm and the recordings of Cz
%garrido;
load garrido_tones

nr_trials=length(garrido_tones);

K=2;
prior=[1 1]';
posterior=[1 1]';

%The prior is the beta distribution with alpha=beta=1
ps=0:0.01:1;

figure()
subplot(4,1,1),plot(ps,betapdf(ps,1,1)),title('Prior');

liklihood=zeros(nr_trials,1);
surprise=zeros(nr_trials,1);
bayesian_surprise=zeros(nr_trials,1);
prediction_error=zeros(nr_trials,1);

for trial=1:nr_trials
    
    prediction=prior(2)/sum(prior);
    prediction_error(trial)=abs(tone-prediction);
    
    %1. Stimulus presentation
    tone=garrido_tones(trial)==550;
    
    %Update the posterior
    posterior(tone+1)=posterior(tone+1)+1;
    
    %2. Compute the liklihood and the surprise
    liklihood(trial)=posterior(tone+1)/sum(prior);
    surprise(trial)=-log2(liklihood(trial));
    
    %Calculate the Bayesian surprise of the current observation
    
    C=exp(gammaln(sum(prior))-gammaln(prior(1))-gammaln(prior(2)));
    C_prime=exp(gammaln(sum(posterior))-gammaln(posterior(1))-gammaln(posterior(2)));
    %psi is the digamma function
    conditional_expectations=psi(sum(posterior))-psi(posterior);
    
    bayesian_surprise(trial)=(log(C_prime/C)-sum((posterior-prior).*conditional_expectations))/log(2);
    
    prior=posterior;
end

save basesian_surprise bayesian_surprise
save surprise surprise

%The learned prior is the beta distribution with alpha=1+n_1, beta=1+n_2
subplot(4,1,2),plot(ps,betapdf(ps,posterior(2),posterior(1))),title('Learned Prior');

subplot(4,1,3),plot(1:nr_trials,surprise),title('Surprise','FontSize',16),
xlabel('Trial','FontSize',12),ylabel('Surprise','FontSize',12)
ax=gca();
axes('position',[0,0,1,1], 'visible','off');
tx1=text('Interpreter','latex', 'Position', [0.15,0.95],'String', 'Surprise Prediction for the Roving Paradigm','FontSize',16);

title('Bayesian Surprise'),xlabel('Trial'),ylabel('Surprise')
subplot(4,1,4), plot(1:50,bayesian_surprise(1:50)),
title('Bayesian Surprise'),xlabel('Trial'),ylabel('Surprise')

print -depsc2 'Surprise Prediction for the Roving Paradigm.eps'