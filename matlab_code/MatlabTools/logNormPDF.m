function lp=logNormPDF(x,mu,sigma)
    lp=-1/2*log(2*pi*sigma^2)-(x-mu)^2/(2*sigma^2);
end