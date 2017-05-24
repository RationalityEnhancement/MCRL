function log_likelihood=logisticLogJoint(y,x,w,alpha)
%alpha is the precision of the Gaussian prior on w, i.e. p(w|alpha)=N(0,diag(1/alpha,...,1/alpha))
%http://www.dcs.gla.ac.uk/~girolami/Machine_Learning_Module_2006/week_4/Lectures/wk_4.pdf
d=length(w);
log_likelihood=dot(y,w'*x)-log(1+exp(w'*x))-1/alpha*dot(w,w)-d/2*log(2*pi*alpha^2);

end