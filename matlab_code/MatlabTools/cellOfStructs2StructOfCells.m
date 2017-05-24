function structOfCells=cellOfStructs2StructOfCells(cellOfStructs)

dataFields=fields(cellOfStructs{1});
nr_structs=length(cellOfStructs);
for f=1:length(dataFields)
    %temp=cell(numel(cellOfStructs{1}.(dataFields{f})),1);
    
    for r=1:nr_structs
        try
            eval(['temp{r}=cellOfStructs{r}.',dataFields{f},';']);
        catch
            disp(['Error during ','temp{r}=cellOfStructs{r}.',dataFields{f},';'])
        end
    end
    eval(['structOfCells.',dataFields{f},'=squeeze(temp);'])
    clear temp
end

end