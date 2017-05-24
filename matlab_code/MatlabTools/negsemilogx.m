function negsemilogx(x,y)
% Do log10 but keep sign
xlog = sign(x).*log10(abs(x));
% Just to get axis limits
plot(xlog,y,'o')
% Get limits
lims = xlim;
wdth = diff(lims);
% Wrap negative data around to positive side
xlog(xlog<0) = xlog(xlog<0) + wdth;
% Plot
plot(xlog,y,'o')
% Mess with ticks
tck = get(gca,'XTick')';
% Shift those that were wrapped from negative to positive (above) back 
% to their original values
tck(tck>lims(2)) = tck(tck>lims(2)) - wdth;
% Convert to string, then remove any midpoint
tcklbl = num2str(tck);
tcklbl(tck==lims(2),:) = ' ';
% Update tick labels
set(gca,'XTickLabel',tcklbl)