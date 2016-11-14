function population = ga()
clear; close all; clc;

%Parameters
popSize = 100;             %population size
mutationRate = 0.01;
sizeTournament = 2;         %tournament size
maxGenerations = 100;
best = Inf;                 %initialise best
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
    
    %crossover
    population2a = population(success(1:2:end),:);
    population2b  = population(success(2:2:end),:);
    Tub = sub2ind(size(population),[1:popSize]',round(1+rand(popSize,1)*(neurons-1)));
    TubA = population2b(Tub)*ones(1,neurons);
    [r,crossover]=find(population2a==TubA); 
    [~,Ord]=sort(r); 
    r = r(Ord); crossover = crossover(Ord); 
    Tub2 = sub2ind(size(population),r,crossover);
    population2a(Tub2) = population2a(Tub);
    population2a(Tub) = population2b(Tub);
    
    %mutation
    index = rand(popSize,1)<mutationRate;
    box1 = sub2ind(size(population2a),1:popSize,round(1+rand(1,popSize)*(neurons-1)));
    box2 = sub2ind(size(population2a),1:popSize,round(1+rand(1,popSize)*(neurons-1)));
    
    box2(index == 0) = box1(index == 0);
    [population2a(box1), population2a(box2)] = deal(population2a(box2), population2a(box1));
    
    %reset
    population = population2a;
end
end