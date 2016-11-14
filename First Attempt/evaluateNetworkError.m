function [regressionError, classificationError] = evaluateNetworkError(dataSets, weights)
%compute output matrix Z with feedForwardArch function
[outputNode, net] = feedForwardArch(dataSets.inputNode, weights, dataSets.bias);
%compute the number of rows and columns in the weights matrix
[rows, outputNumber] = size(weights);
%subtract target output matrix from the outputNode matrix
regressionError = sum(sum((outputNode - dataSets.outputNode).^2)) / (dataSets.count* outputNumber);
%calculate the classNumber from outputNode matrix
classNumber = classFromOutput(outputNode);
%find number of classes that correspond with target classes and normalise
%by dividing by the number of samples (dataSet.count)
classificationError = sum(classNumber ~= dataSets.classNumber) / dataSets.count;

%aim: inequality returns a matrix of 0s and 1s with 1s in matrix positions
%where the corresponding class and target classes are not equal. 
%this needs work