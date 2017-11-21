function [Y, Y_w] = vl_myfc(X, W, dzdy)
%[DZDX, DZDF, DZDB] = vl_myconv(X, F, B, DZDY)
%regular fully connected layer

X_t = zeros(size(X{1},1)^2,length(X));
for ix = 1 : length(X)
    x_t = X{ix};
    X_t(:,ix) = x_t(:);
end
if nargin < 3
    Y = W'*X_t;
else
    Y = W * dzdy;
    Y_w = X_t * dzdy';
end