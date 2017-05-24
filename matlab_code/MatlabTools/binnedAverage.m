function [binned_avg,bin_sem]=binnedAverage(data,bin_width)

nr_points=length(data);
lower_bounds=1:bin_width:nr_points;
upper_bounds=bin_width:bin_width:nr_points;
nr_bins=numel(upper_bounds);

binned_avg=NaN(nr_bins,1);
bin_sem=NaN(nr_bins,1);
for b=1:nr_bins
    binned_avg(b)=nanmean(data(lower_bounds(b):upper_bounds(b)));
    bin_sem(b)=sem(data(lower_bounds(b):upper_bounds(b)));
end

end