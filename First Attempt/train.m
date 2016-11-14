function [weights, errorValues, epochNumber] = train(dataSets)
%attempt to limit maximum number of samples used for training
%nargin returns the number of function input arguments passed in the call
%of this function
if nargin == 0
    clc; %CLear Command window. This gives a blank slate for the ANN to begin
    dataSets = loadDataSet(''); %this needs connecting and sorting
    %construct bias vectors with equal number of rows to the inputs in
    %training, validation and test sets. Must only contain 1s.
    maximumTrainingData = -1;
    if maximumTrainingData > 0
       dataSets.training.inputNode = dataSets.training.inputNode(1:maximumTrainingData,:);
       dataSets.training.outputNode = dataSets.training.outputNode(1:maximumTrainingData,:);
       dataSets.training.classNumber = dataSets.training.classNumber(1:maximumDataSets);
       dataSets.training.bias = dataSets.training.bias(1:maximumDataSets);
       dataSets.training.count = maximumTrainingData;
    end
    
    train(dataSets);
    
    return
end

%debug flag
plotData = true;
maximumWeight = 1/2;
epochNumber = 1;
%maximumIterations = 500;
%learningRate = 0.1;
validationStop = 0.1;

%initialise weight matrix
weights = weightInitialisation(maximumWeight,...
            dataSets.inputCount +1, dataSets.outputCount);

while true
    weights = backPropagationFunc (dataSets.training.inputNode,...
        weights, dataSets.training.bias, learningRate,...
        dataSets.training.outputNode); %possible problem here as bpfunc has targetOutputs not outputNode
    
    [trainingRegressionError(epochNumber),...
        trainingClassificationError(epochNumber)] = ...
        evaluateNetworkError(dataSets.training, weights);
        
    [validationRegressionError(epochNumber),...
        validationClassificationError(epochNumber)] = ...
        evaluateNetworkError(dataSets.validation, weights);
    
    [testRegressionError(epochNumber),...
        testClassificationError(epochNumber)] = ...
        evaluateNetworkError(dataSets.test, weights);
    %mod gives the remainder after division of epochNumber by 10
    %fprintf writes data to text file. 
    %'' surrounds text; \t represents a tab. 
    % %g is to format floating point number data as text. 
    % \n represents a new line 
    if mod(epochNumber, 10) == 0
        fprintf('\tEpochs: %g', epochNumber);
        fprintf('\tTraining: %g (%g)\n', trainingRegressionError(epochNumber), trainingClassificationError(epochNumber));
        fprintf('\tValidation: %g (%g)\n', validationRegressionError(epochNumber), validationClassificationError(epochNumber));
        fprintf('\tTest: %g (%g)\n', testRegressionError(epochNumber), testClassificationError(epochNumber));
        fprintf('\n');
    end
    %maintaing the loop
    if(validationRegressionError(epochNumber)) < validationStop || (epochNumber >= maximumIterations)
        break;
    end
    epochNumber = epochNumber + 1;
end
% needs work - not finished    
    
    
    
    
    
    
    
    
    
    
    
    