function p=pdfOfMaxOfGaussians(x,mu,sigma)

n=numel(mu);

cumulative_distributions=zeros(numel(x),n);
densities=zeros(numel(x),n);
for k=1:n
    densities(:,k)=normpdf((x-mu(k))/sigma(k));
    cumulative_distributions(:,k)=normcdf((x-mu(k))/sigma(k));
end

p=0;
for l=1:n
    others=setdiff(1:n,l);
    p=p+1/sigma(l)*densities(:,l).*prod(cumulative_distributions(:,others),2);
end

p=reshape(p,size(x));

end