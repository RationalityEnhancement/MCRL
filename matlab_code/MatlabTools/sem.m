function standard_error = sem( data,dim )
%sem computes the standard error of the mean

if not(exist('dim','var'))
    dim=1;
end

if all(data(:)==data(1))
    size_vector=size(data);
    size_vector(dim)=1;
    standard_error=zeros(size_vector);
else
    standard_error=nanstd(data,[],dim)./sqrt(sum(not(isnan(data)),dim));
end

end

