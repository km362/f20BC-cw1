function weights = weightInitialisation(maximumWeight, rowNumber, columnNumber)
weights = (-1 +(2*rand(rowNumber, columnNumber))) * maximumWeight;

%use of "rand" here is to generate uniformly distributed random numbers within a range.
%The general formula is r = a + ((b-a)*rand(N,M)).
%a and b are the upper and lower bounds of the range. In this case it is -1 and 1: (1-(-1)) = 2.
%N and M are the the number of rows and coloumns that the final matrix will consist of, respectively. 
%This matrix is multiplied by maximumWeught to make sure it is in the same
%scale 