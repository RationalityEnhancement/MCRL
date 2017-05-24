function samples=mvnrnd_from_Pi(mu,Pi,nr_samples)
%Draw nr_samples from a multi-variate normal distribution with mean mu and
%precision matrix Pi.

%1. Make a Cholesky decomposition of A: A=L'*L.
R=chol(Pi);
samples=zeros(nr_samples,numel(mu));
for s=1:nr_samples
    %2. Choose n independent standard normal variates z=(z1,...,zn)
    z=randn(size(mu,1),1);
    
    %3. The vector v with L*v=z is N(0,A)-distributed. Solve this system of
    %linear equations by backward substitution.
    n = length( z );
    x = zeros( n, 1 );
    for i=n:-1:1
        x(i) = ( z(i) - R(i, :)*x )/R(i, i);
    end
    
    samples(s,:)=mu+x;
end
end