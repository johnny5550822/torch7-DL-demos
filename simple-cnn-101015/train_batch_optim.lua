-- ###############
-- This version has batch processing and function to select different gradient learning method
-- include confusion matrix

-- The training data can be also obtained in https://github.com/hpenedones/luacnn

-- CPU only
-- ###############

require "torch"
require "nn"
require "math"
require "optim"
require 'xlua' -- provides useful tools, such as progress bars
require 'gnuplot'

-- add command line input
cmd = torch.CmdLine()
cmd:text() -- log a custom text message
cmd:text('Training a simple CNN')
cmd:text()
cmd:text('Options')
cmd:option('-lr',0.01,'Learning rate')
cmd:option('-me',2,'Maximum Epochs')
cmd:option('-bs',50,'Batch size')
cmd:option('-optim','adagrad','Optimization method')

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

-- initiate the confusion matrix
-- confusion = optim.ConfusionMatrix(10)

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

-- select optimization algorithm
function select_optim(optimization)
	local optimState -- the state of the method
	local optimMethod --what method to use
	if optimization == 'lbfgs' then
	  optimState = {
	    learningRate = 1e-1,
	    maxIter = 2,
	    nCorrection = 10
	  }
	  optimMethod = optim.lbfgs
	elseif optimization == 'sgd' then
	  optimState = {
	    learningRate = 1e-2,
	    weightDecay = 0.0005,
	    momentum = 0.9,
	    learningRateDecay = 1e-4
	  }
	  optimMethod = optim.sgd
	elseif optimization == 'adagrad' then
	  optimState = {
	    learningRate = 1e-1,
	  }
	  optimMethod = optim.adagrad
	else
	  error('Unknown optimizer')
	end
	return optimState, optimMethod
end


--train a network
function train_network(network,dataset, optimMethod, optimState, parameters, gradParameters)
	print('Training the network......')
	local criterion = nn.ClassNLLCriterion()

	----------------------------- this is the evaluation function for the optimization
	function feval(x)
		-- check if x is updated to date		
		if x ~= parameters then	
			parameters:copy(x)	-- parameters are first defined in the main()
		end

		-- prepare for the batch
		startPos = batch_size * (batch_counter-1) + 1
		endPos = batch_size * batch_counter
		input = torch.Tensor(batch_size,1,patchSize,patchSize):zero() -- this iteration batch
		target = torch.Tensor(batch_size):zero()
		k = 1 -- internal counter for the batch
		for i=startPos,endPos do
			input[{{k},1,{},{}}]:copy(dataset[i][1])
			target[k] = dataset[i][2]
			k = k + 1
		end
		batch_counter = batch_counter + 1 -- update batch_counter

		-- set gradient parameters to zero
		gradParameters:zero()

		------------------- compute loss and gradient
		-- forward propagation
		local batch_outputs = network:forward(input)
		local batch_loss = criterion:forward(batch_outputs,target)
		-- backward propagation
		local dloss_doutput = criterion:backward(batch_outputs,target)
		network:backward(input, dloss_doutput)

		return batch_loss, gradParameters
	end

	----------------------- start the training processing
	local losses = {} -- training losses for each epoch
	for epoch = 1, maxEpochs do
		-- batch processing parameters
    	batch_counter = 1
    	print(string.format('Epoch No:%d. (Max=%d)',epoch,maxEpochs))

		for iteration= 1, maxIterations do
			xlua.progress(iteration,maxIterations) -- progress bar

			local _,minibatch_loss = optimMethod(feval,parameters, optimState)

			-- print loss at certain time; don't have to print all the time
			if iteration%10 ==0 then
				print('Training loss:'..minibatch_loss[1])
			end

			-- update losses for each epoch
			if iteration == maxIterations then
				losses[#losses + 1] = minibatch_loss[1]
			end
		end

		-- collect the garbage in case
		collectgarbage()
	end

	-- once this training is done, plot the losses
	gnuplot.plot({
 		'train-loss',
  		torch.range(1, #losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
  		torch.Tensor(losses),           -- y-coordinates (the training losses)
  	'-'})
  	gnuplot.title('Training error')
	gnuplot.xlabel('Number of epochs')
	gnuplot.ylabel('Loss')
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

------------------------------------------------------------START
--main 
function main()
	local training_dataset, testing_dataset, classes, classes_names = dofile('usps_dataset.lua')
	local network = create_network(#classes)
	s_training_dataset = shuffleData(training_dataset)

    -- determine the optimization method
    local optimState, optimMethod = select_optim(params.optim)
    local parameters, gradParameters = network:getParameters() -- get the parameters of the network

	train_network(network,s_training_dataset, optimMethod, optimState, parameters, gradParameters)
	test_predictor(network, testing_dataset, classes, classes_names)
end

--run 
main()








































