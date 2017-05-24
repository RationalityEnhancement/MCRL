function [p_argmax,sigma_argmax]=distributionOfArgMax(mu,sigma)

nr_samples=1000;
samples=mvnrnd(mu(:),diag(sigma),nr_samples);
p_argmax=histc(argmax(samples'),1:numel(mu))/nr_samples;

for i=1:numel(p_argmax);
    sigma_argmax(i)=sqrt(p_argmax(i)*(1-p_argmax(i)));
end

end