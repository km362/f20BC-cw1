function neuralnet = simpleNN()
% create data set
[Att, Classifier] = exampleData; 

learningRate = 0.1; 
%intended for use when COCO func loaded
%dimension = 2;
numberOfNodes = 10; %2 input 2 output 6 hidden
%below parameters intended for use
%numberofHiddenLayers = 2;
%numberofHiddenNodes = [3;3];
%inRange = [-5 5];
numberOfIterations = 700;
%initialise the NN
neuralnet = 0; 
%create new figure window
figure;
%holds plot in place
hold on;
%returns size of the Array Att
sz = size(Att);

%attempt to load sphere data
%[inputNode, targetOutput] = createTrainingDataSphere(numberOfNodes, dimension, inRange);

% matrices initalisation with random weights range of -1 - +1
V = (-1 +(2*rand(length(Classifier(1,:)),numberOfNodes)));
W = (-1 + (2*rand(numberOfNodes, length(Att(1,:))))); 

%use of "rand" here is to generate uniformly distributed random numbers within a range.
%The general formula is r = a + ((b-a)*rand(N,M)).
%a and b are the upper and lower bounds of the range. In this case it is -1 and 1: (1-(-1)) = 2.
%N and M are the the number of rows and coloumns that the final matrix will consist of, respectively

%Parameters for GA
popSize = 100;             %population size
mutationRate = 0.01;        %mutation Rate
sizeTournament = 2;         %tournament size
maxGenerations = 700;        %max iterations of GA for loop
%best = inf;                 %initialise best
neurons = 60;               %input output and hidden layer neurons

%initialise population
[~,population] = sort(rand(popSize,neurons),2);

for generation = 1:maxGenerations
   %Fitness Function
   fitness = var(diff(population,[],2),[],2);  
   
   %print statistics
    fprintf('Generation: %d    Mean Fitness: %d    Best Fitness: %d\n',generation,round(mean(fitness)),round(max(fitness)));
    
    %tournament selection method
    tournament = round(1+rand((popSize*2),sizeTournament)*(popSize-1));
    [~,index] = max(fitness(tournament),[],2); %index of most successful in tournament
    success = tournament(sub2ind(size(tournament),(1:2*popSize)',index)); %sub2ind: converts subscript to linear indices
    
    %single point crossover - report has details of explanation
    population2a = population(success(1:2:end),:);
    population2b  = population(success(2:2:end),:);
    parent1 = sub2ind(size(population),[1:popSize]',round(1+rand(popSize,1)*(neurons-1)));
    parent2 = population2b(parent1)*ones(1,neurons);
    [r,crossover]=find(population2a==parent2); 
    [~,mix]=sort(r); 
    r = r(mix); crossover = crossover(mix); 
    child = sub2ind(size(population),r,crossover);
    population2a(child) = population2a(parent1);
    population2a(parent1) = population2b(parent1);
    
    %mutation
    index = rand(popSize,1)<mutationRate;
    box1 = sub2ind(size(population2a),1:popSize,round(1+rand(1,popSize)*(neurons-1)));
    box2 = sub2ind(size(population2a),1:popSize,round(1+rand(1,popSize)*(neurons-1)));
    
    box2(index == 0) = box1(index == 0);
    [population2a(box1), population2a(box2)] = deal(population2a(box2), population2a(box1));
    
    %reset
    population = population2a;
end 

while neuralnet < numberOfIterations

    neuralnet = neuralnet + 1; %loop incrementation

    % Iteration through Examples
    for E=1:sz(1)
        % Input data from current example set
        I = Att(E,:).';
        D = Classifier(E,:).';
        
        hiddenWeights = activationFunc(W*I); %creates weights throughout netowkr
        outputWeights = activationFunc(V*hiddenWeights); % creates weights through network
        outputError = outputWeights.*(1-outputWeights).*(D-outputWeights); %outputerror
        nodeHiddenError = hiddenWeights.*(1-hiddenWeights).*(V.'*outputError); %calculates error in layer before output layer, ie hidden layers
        %adjust weights ready for the net increment of the loop
        V = V + learningRate.*outputError*(hiddenWeights.');
        W = W + learningRate.*nodeHiddenError*(I.');
    end
    
  % Calculate RMS error
    RMSerror = 0;
    for E=1:sz(1)
        D = Classifier(E,:).';
        I = Att(E,:).';
        RMSerror = RMSerror + norm(D-activationFunc(V*activationFunc(W*I)),2);
    end
    
    y = RMSerror/sz(1);
    plot(neuralnet,log(y),'*');

end
end

function x = activationFunc(x)
x = (tanh(x)+1)/2;
%takes in the matrix X and returns the matrix Y
end

function y = activationDerivative(x)
y = (1-tanh(x).^2)/2;
%take in matrix X and return matrix Y
end