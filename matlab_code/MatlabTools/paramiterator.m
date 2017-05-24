classdef paramiterator
% Paramiterator
%
% Flattens arbitrarily nested loops into a single loop, iterating 
% over all combinations of loop parameters.
%
% Makes it very easy to change the iteration parameters without rewriting 
% loops, reindenting code, updating indices, etc.
%
% Very convenient for simulation problems where the parameters being tested
% frequently change, but the computation stays the same.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Sample nested loop code like this:
%
% avalues = [1 2]; bvalues = [10 20]; cvalues = [100 200 300];
% for ai = 1:length(avalues)
%     for bi = 1:length(bvalues)
%         for ci = 1:length(cvalues)
%             scoretable(ai,bi,ci) = avalues(ai) + bvalues(bi) + cvalues(ci);
%         end
%     end
% end
% 
%   Can be flattened into the following, and any parameter can be added or 
%   removed by changing only paramvariables and paramvalues:
%
% paramvariables = {'A' 'B' 'C'};
% paramvalues = { {1 2} {10 20} {100 200 300}};
% piter = paramiterator(paramvariables,paramvalues);
% scorelist = zeros(length(piter),1);
% for i = 1:length(piter)
%     setvalues(piter,i);
%     scorelist(i) = A + B + C;
% end
% scoretable = listtotable(piter,scorelist);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature 1: Display a legible table of results
% (Note: displaytable must be downloaded separately from
%   http://www.mathworks.com/matlabcentral/fileexchange/27920-displaytable)
% 
% paramnames = {'Aardvark' 'Bison' 'Cat'};
% displaytable(scoretable,paramnames,paramvalues); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature 2: Parameter pairing
% 
% When parameters need to be paired and their values assigned in sync with each other, they can
% be put into a single list, e.g. 
%   paramvariables = {'A' {'B' 'C'}};
%   paramvalues = { {1 2} {{10 100} {20 200}}};
% will independently iterate over A and (B and C together), not over A, B, and C.
% First, B=10 and C=100; next, B=20 and C=200.
%
% Helper function paramiterator.pair can help prepare variables for this process.
% For example:
%   {{10 100} {20 200}} = paramiterator.pair({{10 20} {100 200}})
%
% Helper function paramiterator.unpairfordisplay prepares a list of paired paramvalues for displaytable.
% For example:
%   displaytable(scoretable,paramvariables,paramiterator.unpairfordisplay(paramvalues));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature 3: Only update variables when necessary 
%   If you have a slow computation that only needs to be updated when one of
%   the parameters changes, make that parameter the leftmost (changes
%   slowest), and only update it when valuechanged(piter,i,variable) is
%   true. Example:
%
% for i = 1:length(piter)
%     ...
%     if valuechanged(piter,i,'A')
%       A = slowcomputation(A);
%     end
%     ...
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Matt Caywood
%
% 2010.07.29 - 0.1 initial release 
% 2010.08.05 - 0.2 added value change test
% 2012.08.03 - 0.3 added paired parameters

properties (SetAccess = private)
   paramvariables
   paramvalues
   settingslist
   numparameters
end 
 

methods(Static)
    %% when multiple parameter values are used, unpair a cell list of cells for display
    function displayparamvalues = unpairfordisplay(paramvalues)

        displayparamvalues = cell(1,length(paramvalues));
        for v = 1:length(displayparamvalues)
            if iscell(paramvalues{v}) && iscell(paramvalues{v}{1})
                for pair = 1:length(paramvalues{v})
                    displayparamvalues{v} = [displayparamvalues{v} paramvalues{v}{pair}{1}];
                end
            else
                displayparamvalues{v} = paramvalues{v};
            end
        end
    end
    
    function pairedparamvalues = pair(paramvalues)
        pairedparamvalues = cell(1,length(paramvalues{1}));
        for pair = 1:length(paramvalues{1})
            val = cell(1,length(paramvalues));
            for v = 1:length(paramvalues)
                val{v} = paramvalues{v}{pair};
            end       
            pairedparamvalues{pair} = val;
        end
    end
end    

methods
   function piter = paramiterator(paramvariables,paramvalues)
        % paramvariables = cell array of strings, e.g. {'A' 'B' 'C'};
        % paramvalues = cell array of cell arrays, e.g. { {1 2} {10 20} {100 200 300}};

        assert(length(piter.paramvariables) == length(piter.paramvalues),'Variables and values must be same length');

        piter.paramvariables = paramvariables;
        piter.paramvalues = paramvalues;

        piter.settingslist = parameterstolist(piter.paramvalues);
        piter.numparameters = size(piter.settingslist,2);

    end

    function l = length(piter)
        % length(piter) returns the length of the list of settings to be iterated over
        l = size(piter.settingslist,1);
    end

    function table = listtotable(piter,scorelist)
        % listtotable(piter,list) converts a list of values derived from 
        % iterating over each combination of parameters in the settings list, 
        % back to an n-dimensional table indexed by parameter.

        if (piter.numparameters == 1)
            table = scorelist;
        else
            parameterlengths = zeros(1,piter.numparameters);
            for i = 1:piter.numparameters
                parameterlengths(i) = length(piter.paramvalues{i});
            end
            table = permute(reshape(scorelist,fliplr(parameterlengths)),piter.numparameters:-1:1);
        end
    end

    function setvalues(piter,i)
        % setvalues(piter,idx) sets all parameters to the values given by
        % entry idx in the settings list.
        for j = 1:piter.numparameters
            % make a list if single variable only
            vars = piter.paramvariables{j};
            vals = piter.settingslist{i,j};
            if ~iscell(vars)
                vars = {vars};
                vals = {vals};
            end
            for v = 1:length(vars)
                evalin('caller',sprintf('%s = %f;',vars{v},vals{v}));
                % MSC seems simpler but not always reliable
                % assignin('caller',piter.paramvariables{j},piter.settingslist{i,j});
            end
        end
    end
    
    function changed = valuechanged(piter,i,variable)
        % valuechanged(piter,i,variable) returns 1 if the variable has changed
        % from last iteration i-1 to iteration i, 0 if it has not.
        flattenedparamvariables = [piter.paramvariables{:}]; % necessary for paired variables
        j = find(ismember(flattenedparamvariables,variable) == 1);

        assert ((i >= 1) && (i <= length(piter)),'invalid iteration');
        assert (~isempty(j),'invalid variable');
        
        if (i <= 1), changed = 1; % first iteration
        elseif ~isequal(piter.settingslist(i,j),piter.settingslist(i-1,j)), changed = 1;
        else changed = 0;
        end
    end

end

end

%% returns a cell list of parameter values
function list = parameterstolist(paramvalues)

    list = recursegenerateparameters(1,paramvalues,{});

    function returnedparams = recursegenerateparameters(pidx,paramvalues,currentparams)
    % 
    % returns array containing all combinations of the parameters
    % 
    % pidx = index of current parameter in paramvalues
    % paramvalues = a cell array containing a bunch of parameter sets
    % currentparams: collects all generated parameter sets

    if (pidx > length(paramvalues))
        % base case: done with all paramvalues, so return
        returnedparams = currentparams;
    else
        % add each new value, recurse, concatenate result to current list
        returnedparams = {};
        newvalues = paramvalues{pidx};
        for nval = newvalues
            returnedparams = [returnedparams ; recursegenerateparameters(pidx+1,paramvalues,[currentparams nval])]; %#ok
        end
    end

    end
end
