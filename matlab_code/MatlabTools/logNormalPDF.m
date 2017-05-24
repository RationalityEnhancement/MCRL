function log_likelihood=logNormalPDF(x,mu,precision_matrix)

    k=length(mu);
    log_likelihood=-k/2*log(2*pi)+1/2*logdet(precision_matrix)-...
        1/2*(x-mu)'*precision_matrix*(x-mu);

end