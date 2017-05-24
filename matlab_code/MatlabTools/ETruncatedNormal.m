function EV=ETruncatedNormal(mu,sigma,x_min,x_max)
    
int_below=normcdf(x_min,mu,sigma).*max(-realmax,x_min);
int_above=(1-normcdf(x_max,mu,sigma)).*min(realmax,x_max);

    int_middle=mu*(normcdf(x_max,mu,sigma)-normcdf(x_min,mu,sigma))-...
        sigma^2*(normpdf(x_max,mu,sigma)-normpdf(x_min,mu,sigma));
    
    EV=int_below+int_middle+int_above;
    
end