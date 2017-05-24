function v=offdiag(A)
%returns a vector with the offdiagonal elements of A
    [n1,n2]=size(A);
    N=numel(A);
    diag_indices=((0:(n1-1))*n1)+(1:n1);
    v=A(setdiff(1:N,diag_indices));
end