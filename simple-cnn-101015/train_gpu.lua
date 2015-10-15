-- ###############

-- a version with GPU support. Assuming your gpu is working properly. Later I may release a version that check your gpu as well

-- This is a minimal implementation (e.g., no batch processing) of a CNN on digit (0-9) classification. This code is largely obtained from luacnn, https://github.com/hpenedones/luacnn
-- The training data can be also obtained in https://github.com/hpenedones/luacnn

-- CPU only
-- ###############

require "torch"
require "nn"
require "math"
require "cutorch" -- for GPU, Cuda
require "cunn" -- for GPU, Cuda

-- add command line input
cmd = torch.CmdLine()
cmd:text() -- log a custom text message
cmd:text('Training a simple CNN')
cmd:text()
cmd:text('Options')
cmd:option('-lr',0.01,'Learning rate')
cmd:option('-maxI',100000,'Maximum Iterations')
cmd:text()

-- parse the input params
params =cmd:parse(arg)

-- variables
learningRate = params.lr
maxIterations = params.maxI

-- initiate gpu support
cutorch.setDevice(1) -- gpu device # 1
cutorch.manualSeed(123) -- seed 

-- create the neural network
function create_network(nb_outputs)
	local cnn = nn.Sequential();	

	-- first convolution, non-linear, and pooling
	cnn:add(nn.SpatialConvolution(1,10,3,3,1,1,0,0)) -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]). E.g., input:1x16x16, beomes 10x14x14
	cnn:add(nn.ReLU()) -- non-linear layer
	cnn:add(nn.SpatialMaxPooling(2,2)) -- becomes 10x7x7

	-- second convolution, non-linear, and pooling
	cnn:add(nn.SpatialConvolution(10,40,2,2,1,1,0,0)) -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]). E.g., input:10x14x14, beomes 40x6x6
	cnn:add(nn.ReLU()) -- non-linear layer
	cnn:add(nn.SpatialMaxPooling(2,2)) -- becomes 40x3x3

	cnn:add(nn.Reshape(40*3*3))
	cnn:add(nn.Linear(40*3*3,nb_outputs))
	cnn:add(nn.LogSoftMax())

	return cnn:cuda()
end

--train a network
function train_network(network,dataset)
	print('Training the network......')
	local criterion = nn.ClassNLLCriterion():cuda()

	for iteration= 1, maxIterations do
		print(string.format('Iteration(max=%d) No.%d',maxIterations,iteration))

		local index = math.random(dataset:size()) --pick example at random
		local input = dataset[index][1] -- size 1x16x16
		local output = dataset[index][2] -- size 1

		-- DEBUG: To check the network output size
		--print(network:forward(input))
		--os.exit()

		-- for cuda
		input = input:float():cuda()

		output = torch.Tensor(1):fill(output):float():cuda()

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

               input = input:float():cuda()
        
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
	-- get the data
	local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
	local network = create_network(#classes)

	-- Timer
	timer = torch.Timer()

	train_network(network,training_dataset)
	test_predictor(network, testing_dataset, classes, classes_names)

	-- end timer
	print("Total time for training(s):%s",timer:time().real)
end

--run 
main()








































