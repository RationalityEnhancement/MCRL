function rmpathsub(directory)
% RMPATHSUB Remove all subdirectories from MATLAB path.
% RMPATHSUB(DIRECTORY) removes DIRECTORY and all
% its subdirectories from the MATLAB search path.

% get path as long string
p=path;

% divide string to directories, don't
% forget the first or the last...
delim=[0 strfind(p, ';') length(p)+1];

for i=2:length(delim)
    direc = p(delim(i-1)+1:delim(i)-1);
    if strncmpi(direc, directory, length(direc))
        rmpath(direc);
    end
end