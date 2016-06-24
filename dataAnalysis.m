function dataAnalysis()

obs = [68.5,81,64,43,53,41.5,47.5,52.5,57.5,62.5,97.5,103,96,82.5,97.5,87.5]';
calc = [73.97,83.86,67.02,42.48,52.92,40.28,47.97,52,58.04,62.8,95.95,206.91,95.95,82.03,100.34,86.06]';
format long;
[obs,idx] = sort(obs);
calc = calc(idx);


n = length(obs);
MSE = sum(sqrt(abs(calc-obs).^2))/n
err = abs(calc-obs)./obs.*100;
b1 = obs\calc
linreg = b1*obs;




close all;
figure;
subplot 121;
h = stem(obs,calc,'filled','LineStyle','none', ...
    'MarkerSize', 6,'MarkerFaceColor','b','LineWidth',2,'MarkerEdgeColor','k'); hold on;

plot(obs,linreg); hold off;
grid on;
title('Observing Error Over Varying Frequencies');
ylabel('Calculated RPM');
xlabel('Observed RPM');
ax = gca;
ax.TickDir = 'out';
ax.XLim = [0 250];
ax.YLim = [0 250];

subplot 122;

g = stem(obs,err);
ylabel('Error %');
xlabel('Observed RPM');
mstr = ['MSE:',num2str(MSE)];
str = {'Error is high','at higher RPMs.'};
text(65,60,str);




end