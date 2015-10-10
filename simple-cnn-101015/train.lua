-- ###############
-- This is a minimal implementation (e.g., no batch processing) of a CNN on digit (0-9) classification. This code is largely obtained from luacnn, https://github.com/hpenedones/luacnn
-- The training data can be also obtained in https://github.com/hpenedones/luacnn

-- ###############

require "torch"
require "nn"
require "math"

-- variables
learningRate = 0.01
maxIterations = 100000

-- create the neural network
function create_network(nb_outputs)
	local cnn = nn.Sequential();	

	cnn:add(nn.SpatialConvolution(1,6,5,5)) -- becomes 12x12x6
	cnn:add(nn.ReLU()) -- non-linear layer
	cnn:add(nn.SpatioalSubSampling(6,2,2,2,2)) -- becomes 6x6x6

	cnn:add(nn.Reshape(6*6*6))
	cnn:add(nn.Linear(6*6*6,nb_outputs))
	cnn:add(nn.LogSoftMax())

	return cnn
end

--train a network
function train_network(network,dataset)
	print('Training the network......')
	local criterion = nn.ClassNLLCriterion()

	for iteration= 1, maxIterations do
		local index = math.random(dataset:size()) --pick example at random
		local input = dataset[index][1]
		local output = dataset[index][2]

		-- forward propagation
		criterion:forward(network:forward(input),output)
		-- zero the accumlated gradient
		network:zeroGradParameters()
		-- backward propagation
		networkbackward(input,criterion:backward(network.output,output))
		-- after backward propagation; update the parameters
		network:updateParameters(learningRate)
	end
end

--main 
function main()
	local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
	local network = create_network(#classes)
	train_network(network,training_dataset)
end

--run 
main()








































