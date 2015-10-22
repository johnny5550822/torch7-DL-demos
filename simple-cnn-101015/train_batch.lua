-- ###############
-- This version is built on top of train.lua, which supports batch processing
-- The training data can be also obtained in https://github.com/hpenedones/luacnn

-- CPU only
-- ###############

require "torch"
require "nn"
require "math"

-- add command line input
cmd = torch.CmdLine()
cmd:text() -- log a custom text message
cmd:text('Training a simple CNN')
cmd:text()
cmd:text('Options')
cmd:option('-lr',0.01,'Learning rate')
cmd:option('-me',10,'Maximum Epochs')
cmd:option('-bs',50,'Batch size')

cmd:text()

-- parse the input params
params =cmd:parse(arg)

-- variables
learningRate = params.lr
maxEpochs = params.me
batch_size = params.bs	
totalImages = 10000 -- we know there are in total 10,000 images; each with size of 1 x 16 x 16
patchSize = 16
maxIterations = totalImages/batch_size -- this is per epochs; I did not calculate the max. total iterations for clarity

-- create the neural network
function create_network(nb_outputs)
	local cnn = nn.Sequential();	

	-- first convolution, non-linear, and pooling
	cnn:add(nn.SpatialConvolution(1,100,3,3,1,1,0,0)) -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]). E.g., input:1x16x16, beomes 10x14x14
	cnn:add(nn.ReLU()) -- non-linear layer
	cnn:add(nn.SpatialMaxPooling(2,2)) -- becomes 10x7x7

	-- second convolution, non-linear, and pooling
	cnn:add(nn.SpatialConvolution(100,400,2,2,1,1,0,0)) -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]). E.g., input:10x14x14, beomes 40x6x6
	cnn:add(nn.ReLU()) -- non-linear layer
	cnn:add(nn.SpatialMaxPooling(2,2)) -- becomes 40x3x3

	cnn:add(nn.Reshape(400*3*3))
	cnn:add(nn.Linear(400*3*3,nb_outputs))
	cnn:add(nn.LogSoftMax())

	return cnn
end

--train a network
function train_network(network,dataset)
	print('Training the network......')
	local criterion = nn.ClassNLLCriterion()

	-- loop
	for epoch = 1, maxEpochs do
		-- batch processing parameters
    	batch_counter = 1

		for iteration= 1, maxIterations do
			print(string.format('Epoch No:%d. Iteration(max=%d) No.%d',epoch,maxIterations,iteration))

			--[[
			local index = math.random(dataset:size()) --pick example at random
			local input = dataset[index][1] -- size 1x16x16
			local output = dataset[index][2] -- size 1
			]]

			-- prepare for the batch
			startPos = batch_size * (batch_counter-1) + 1
			endPos = batch_size * batch_counter
			input = torch.Tensor(batch_size,1,patchSize,patchSize):zero() -- this iteration batch
			output = torch.Tensor(batch_size):zero()
			k = 1 -- internal counter for the batch
			for i=startPos,endPos do
				input[{{k},1,{},{}}]:copy(dataset[i][1])
				output[k] = dataset[i][2]
				k = k + 1
			end
			batch_counter = batch_counter + 1 -- update batch_counter

			-- forward propagation
			criterion:forward(network:forward(input),output)
			-- zero the accumlated gradient
			network:zeroGradParameters()
			-- backward propagation
			network:backward(input,criterion:backward(network.output,output))
			-- after backward propagation; update the parameters
			network:updateParameters(learningRate)
		end
		print(batch_counter)
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

-- shuffle the data
function shuffleData(data)
	local newOrder = torch.randperm(data:size(1))
	local shuffle_data = {}
	for i = 1,data:size(1) do
		shuffle_data[i] = data[newOrder[i]]
	end

	return shuffle_data
end

--main 
function main()
	local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
	local network = create_network(#classes)
	s_training_dataset = shuffleData(training_dataset)

	train_network(network,s_training_dataset)
	test_predictor(network, testing_dataset, classes, classes_names)
end

--run 
main()








































