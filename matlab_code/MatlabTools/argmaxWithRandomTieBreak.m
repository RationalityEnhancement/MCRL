function choice=argmaxWithRandomTieBreak(values)

maximum=max(values);
best_actions=find(values==maximum);
choice=draw(best_actions);

end