function outputNode = outputFromClass(classNumber)
sampleNumber = length(classNumber);
outputNode = zeros(sampleNumber, max(classNumber));
for i = 1:sampleNumber
    outputNode(m, classNumber(m)) = 1;
end