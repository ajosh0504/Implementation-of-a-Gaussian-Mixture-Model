function [ y ] = gauss_dist(x,meu,sigma )
%GAUSS_DIST function for gaussian distribution
    y=(1/(sigma*sqrt(2*pi)))*exp((-(x-meu).^2)./(2*sigma^2));
end
