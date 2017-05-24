function filtered=selectFromStruct(unfiltered,criterion)

fields=fieldnames(unfiltered);

for f=1:numel(fields)
    fieldname=fields{f}; 
    if size(unfiltered.(fieldname),1)==numel(criterion)
        filtered.(fieldname)=unfiltered.(fieldname)(criterion,:);    
    elseif size(unfiltered.(fieldname),2)==numel(criterion)
        filtered.(fieldname)=unfiltered.(fieldname)(:,criterion);    
    end
end

end