-- ###############
-- This is a minimal implementation (e.g., no batch processing) of a CNN on digit (0-9) classification. This code is largely obtained from luacnn, https://github.com/hpenedones/luacnn
-- The training data can be also obtained in https://github.com/hpenedones/luacnn

-- CPU only
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
	cnn:add(nn.SpatialSubSampling(6,2,2,2,2)) -- becomes 6x6x6

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
		print(string.format('Iteration(max=%d) No.%d',maxIterations,iteration))

		local index = math.random(dataset:size()) --pick example at random
		local input = dataset[index][1]
		local output = dataset[index][2]

		-- forward propagation
		criterion:forward(network:forward(input),output)
		-- zero the accumlated gradient
		network:zeroGradParameters()
		-- backward propagation
		network:backward(input,criterion:backward(network.output,output))
		-- after backward propagation; update the parameters
		network:updateParameters(learningRate)
	end
end

--test the network
function test_predictor(predictor, test_dataset, classes, classes_names)

        local mistakes = 0
        local tested_samples = 0
        
        print( "----------------------" )
        print( "Index Label Prediction" )
        for i=1,test_dataset:size() do

               local input  = test_dataset[i][1]
               local class_id = test_dataset[i][2]
        
               local responses_per_class  =  predictor:forward(input) 
               local probabilites_per_class = torch.exp(responses_per_class)
               local probability, prediction = torch.max(probabilites_per_class, 1) 

                      
               if prediction[1] ~= class_id then
                      mistakes = mistakes + 1
                      local label = classes_names[ classes[class_id] ]
                      local predicted_label = classes_names[ classes[prediction[1] ] ]
                      print(i , label , predicted_label )
               end

               tested_samples = tested_samples + 1
        end

        local test_err = mistakes/tested_samples
        print ( "Test error " .. test_err .. " ( " .. mistakes .. " out of " .. tested_samples .. " )")

end

--main 
function main()
	local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
	local network = create_network(#classes)

	train_network(network,training_dataset)
	test_predictor(network, testing_dataset, classes, classes_names)
end

--run 
main()








































