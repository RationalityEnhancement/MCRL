function mll=marginalLogLikelihoodDirichletMultinomial(observations,alphas)
%logaritm of marginal likelihood of a Dirichlet-Multinomial distribution
%according to Michael Jordan's lecture notes: https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&ved=0CDQQFjAA&url=http%3A%2F%2Fwww.cs.berkeley.edu%2F~jordan%2Fcourses%2F281B-spring01%2Flectures%2Fmarginal.ps&ei=ps7yUImyIa764QTPg4GoCQ&usg=AFQjCNGsFW9F9ZDoQw9xa7nc7Cjz8ufXzg&bvm=bv.1357700187,d.bGE

%{
    observations: sequence of multinomial realizations coded as a vector of
    numbers in {1,...,N}.
    alphas: parameter vector of the Dirichlet prior on the outcome
    probabilities
%}

%1. convert observations into counts
N=length(alphas);
[counts values]=hist(observations,(1:N)');

%2. apply formula
mll=gammaln(sum(alphas))-gammaln(sum(alphas+counts'))+...
    sum(gammaln(alphas+counts')-gammaln(alphas));

end