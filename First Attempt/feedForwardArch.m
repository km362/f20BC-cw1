function [outputNode, net] = feedForwardArch(inputNode, weights, bias)
net = [inputNode, bias] * weights;
outputNode = activationFunc(net);
end