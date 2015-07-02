function contourPlot(func,trueValue,Wmle)

[x,y] = meshgrid(-3:0.05:3,-3:0.05:3); % Create grid.
[r,c]=size(x);

data = [x(:) y(:)];

p = func(data);
p = reshape(p, r, c);

contourf(x,y,p,256,'LineColor','none');
colormap(jet(256));
axis square;
 
xlabel(' W0 ');
ylabel(' W1 ','Rotation',0);
if(length(trueValue) == 2)
    hold on;
    plot(trueValue(1),trueValue(2),'+');
end

hold on;
plot(Wmle(1),Wmle(2),'o');

% hold on;
% plot(postMunew(1),postMunew,'*');

end