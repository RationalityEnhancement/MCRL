function date_vector=parseDateVector(time_string)

if not(isempty(findstr('T',time_string)))
    separator='T';
else
    separator=' ';
end

date_and_time=strsplit(time_string,separator);
year_month_day=strsplit(date_and_time{1},'-');
hour_min_sec=strsplit(date_and_time{2}(1:8),':');
date_vector=cellfun(@(str) str2double(str), {year_month_day{:},hour_min_sec{:}}')';

end