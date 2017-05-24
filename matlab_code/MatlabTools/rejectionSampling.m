function samples=rejectionSampling(propose,q,p,M,nr_samples)
%samples=rejectionSampling(propose,q,p,M,nr_samples)
%Rejection sampling from P.
%propose: samples from Q
%q: proposal density
%p: target density
%M: normalization constant such that M*q(x)>=p(x) for all x

nr_proposals=ceil(10*max(1,M)*nr_samples);

nr_batches=ceil(nr_proposals/1e6);

for i=1:nr_batches
    proposals=reshape(propose(nr_proposals),[nr_proposals,1]);
    p_accept=p(proposals)./(M*q(proposals));
    accepted=rand(nr_proposals,1)<=p_accept;
    samples_by_batch{i,1}=proposals(accepted,1);
    if sum(accepted)>nr_samples
        samples_by_batch{i,1}(1:nr_samples)=samples_by_batch{i,1}(1:nr_samples);  
    end
end

samples=cell2mat(samples_by_batch);
samples=samples(1:nr_samples);

end