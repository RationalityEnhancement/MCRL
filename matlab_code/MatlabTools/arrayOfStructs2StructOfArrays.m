function structOfArrays=arrayOfStructs2StructOfArrays(arrayOfStructs)

dataFields=fields(arrayOfStructs);
nr_structs=length(arrayOfStructs);
for f=1:length(dataFields)
    
    s=size(arrayOfStructs(1).(dataFields{f}));
    s_combined=[nr_structs,s];
    n_dims=length(s);
    

    eval(['temp=NaN(s_combined);']);
    
    for r=1:nr_structs
        if n_dims==1
            eval(['temp(r,:)=arrayOfStructs(r).',dataFields{f},';']);
        elseif n_dims==2
            eval(['temp(r,:,:)=arrayOfStructs(r).',dataFields{f},';']);
        elseif n_dims==3
            eval(['temp(r,:,:,:)=arrayOfStructs(r).',dataFields{f},';']);
        end        
    end
    eval(['structOfArrays.',dataFields{f},'.values=squeeze(temp);'])
    eval(['structOfArrays.',dataFields{f},'.mean=squeeze(mean(temp));'])
    eval(['structOfArrays.',dataFields{f},'.var=squeeze(var(temp));'])
    eval(['structOfArrays.',dataFields{f},'.sem=squeeze(std(temp))/sqrt(nr_structs);'])
    eval(['structOfArrays.',dataFields{f},'.CI_high=structOfArrays.',dataFields{f},'.mean+squeeze(1.96*std(temp)/sqrt(nr_structs));'])
    eval(['structOfArrays.',dataFields{f},'.CI_low=structOfArrays.',dataFields{f},'.mean-squeeze(1.96*std(temp)/sqrt(nr_structs));'])
    clear temp
end

end