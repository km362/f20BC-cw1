function y = randInt(m, n, range)
%floor rounds element down to the nearest integer. Used floor instead of
%round because floor converts logical and char elements into double values,
%whilst round needs the elements to be a single or double. 
y = floor(rand(m,n)*range)