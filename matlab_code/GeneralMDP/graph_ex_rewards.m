fig = figure();
for n=1:numel(num_arms)
    plot(1:10,mu(n,1:10),'LineWidth',3); hold on;
end

set(gca,'FontSize',16,'XScale','log')
xlabel('Horizon','FontSize',18)
ylabel('Expected Return (95% Confidence Intervals)','FontSize',18)
legend('2 arms','3 arms','4 arms','5 arms','6 arms','location','Southeast');
saveas(fig,['../results/figures/GeneralMDPHorizonSimulations', int2str(c), '.fig'])
saveas(fig,['../results/figures/GeneralMDPHorizonSimulations', int2str(c), '.png'])