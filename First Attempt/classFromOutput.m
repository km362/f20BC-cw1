function classNumber = classFromOutput(outputNode)
[values, classNumber] = max(outputNode, [], 2);
end
%needs word