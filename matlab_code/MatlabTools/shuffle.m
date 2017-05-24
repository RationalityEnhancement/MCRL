function array=shuffle(array)
%shuffles the elements of an array into a random order
    N=numel(array);
    order=randperm(N);
    array=array(order);
end