function [E,sigma]= EVMaxBeta(alphas, betas)

nr_samples = 1000;

alphas = reshape(alphas,[1,numel(alphas)]);
betas = reshape(betas,[1,numel(betas)]);
ps= betarnd(repmat(alphas,[nr_samples,1]),repmat(betas,[nr_samples,1]));

max_ps = max(ps');

E = mean(max_ps);
sigma = std(max_ps)/sqrt(nr_samples);

end