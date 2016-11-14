function weights = backpropagationFunc(inputNode, weights, bias, learningRate, targetOutput)
[sampleNumber cols] = size(inputNode);
sampleIndex = randInt(1, 1, sampleNumber) +1;

[outputNode, net] = feedForwardArch(inputNode(sampleIndex,:), weights, bias(sampleIndex));
error = targetOutput(sampleIndex,:)- outputNode;
delta = error.*activationDervivative(net);
deltaWeights = learningRate*kron([inputNode(sampleIndex,:), bias(sampleIndex)], delta);
weights = deltaWeights + weights;

end