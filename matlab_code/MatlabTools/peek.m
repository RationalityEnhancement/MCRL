function varargout = peek(varargin)
% PEEK - a simple evaluation monitor utility
%
% Usage:
%   peek()
%   [p1,...] = peek()
%   peek(p1,...)
%   [p1,...] = peek(p1,...)
%
% Description:
%   PEEK is a general-purpose evaluation monitor that simply records and outputs
%   any values passed to it. When called with one or more input arguments, 
%   PEEK records the values of the inputs and passes them directly to the
%   outputs. When called with no input values, PEEK outputs the recorded values
%   as cell arrays and clears the record.
%
% Examples (see also PEEKTEST conveniently bookmarked for you by M-LINT):
%
%   peek(); % clear the log
%   f = @(x)sin(x);
%   fpeek = @(x)f(peek(x));
%   quad(fpeek,0,1);
%   peek() % see which x values were used in each step of quad
%

% Author:
%   Ben Petschel 29/10/2013
%
% Version history:
%   29/10/2013 - initial version
%

persistent npeek peekval

if isempty(npeek)
  npeek = 0;
  peekval = {};
end

if nargin == 0
  % return the stored values and reset the record
  for i=1:size(peekval,2)
    varargout{i} = peekval(1:npeek,i);
  end
  npeek = 0;
  peekval = {};
else
  % pass all inputs to the output and save a copy in peekval
  varargout = varargin;
  npeek = npeek + 1;
  if npeek == 1
    peekval = varargin;
  else
    if npeek > size(peekval,1)
      % use size-doubling for efficiency
      peekval = [peekval;cell(size(peekval))];
    end
    peekval(npeek,:) = varargin;
  end % if npeek==1 ... else ...
end % if nargin==0 ... else ...

end % main function peek(...)


function testpeek()
% test function - step through these examples in cell mode

%% common data
x0 = 0;
x1 = pi;
f = @(x)sin(x);


%% example 1: save x values passed to a function

peek(); % clear the log (not necessary but good practice)
fpeek = @(x)f(peek(x));
quad(fpeek,x0,x1);
peek() % see which x values were used in each step of quad

%% compare with values used by integrate (ignore the error if your version doesn't have INTEGRATE)
try
  integral(fpeek,x0,x1);
  peek()
catch %#ok<CTCH>
end

%% more complicated example: save both x and f(x)
fpeek = @(x)peek(f(x),x); % quad will only use the first value f(x)
quad(fpeek,x0,x1);
[fq,xq] = peek(); % see which values of f(x) and x were used by quad
try
  integral(fpeek,x0,x1);
  [fi,xi] = peek(); % see which values of f(x) and x were used by integrate
catch %#ok<CTCH>
end

%% plot the points used by quad() and integrate()
for i=1:numel(xq)
  plot(xq{i},fq{i},'bx');
  hold all
  text(xq{i},fq{i},num2str(i),'HorizontalAlignment','Right','VerticalAlignment','Bottom');
end
try
  for i=1:numel(xi)
    plot(xq{i},fq{i},'rx');
    hold all
    text(xi{i},fi{i},num2str(i),'HorizontalAlignment','Right','VerticalAlignment','Bottom','Color','red');
  end
catch %#ok<CTCH>
end

end % test function peektest(...)
