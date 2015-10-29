-- ###############
-- gpu version
-- with optim selection
-- batch
-- This is the main

-- ###############

require "torch"
require "nn"
require "math"
require "optim"
require 'xlua' -- provides useful tools, such as progress bars
require 'gnuplot'
require 'cutorch'
require 'cunn'
require 'paths'

-- my own modality
require './model/simple_model.lua' -- one way to incorporate external function

-- parse command line input
local opts = paths.dofile('opts.lua')
params = opts.parse(arg)

-- variables
learningRate = params.lr
maxEpochs = params.me
batch_size = params.bs	
totalImages = 10000 -- we know there are in total 10,000 images; each with size of 1 x 16 x 16
patchSize = 16
maxIterations = totalImages/batch_size -- this is per epochs; I did not calculate the max. total iterations for clarity

-- initiate gpu support
cutorch.setDevice(1) -- gpu device # 1
cutorch.manualSeed(123) -- seed 

-- initiate the confusion matrix to store tp, tn, etc
local train_confusion = optim.ConfusionMatrix(10) 

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
	elseif optimization == 'adadelta' then
	  optimState = {
	  }
	  optimMethod = optim.adadelta
	else
	  error('Unknown optimizer')
	end
	return optimState, optimMethod
end


--train a network. You may not like this function because there are many inputs. Since I intend to use function to separate different parts, this is one way to go. I don't like it too much too. I will try to improve it later
function train_network(network,dataset, optimMethod, optimState, parameters, gradParameters, testing_dataset, classes, classes_names)
	print('Training the network......')
	local criterion = nn.ClassNLLCriterion():cuda()

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

		-- cuda
		input = input:float():cuda()
		target = target:float():cuda()

		-- set gradient parameters to zero
		gradParameters:zero()

		------------------- compute loss and gradient
		-- forward propagation
		local batch_outputs = network:forward(input)
		local batch_loss = criterion:forward(batch_outputs,target)

		-- backward propagation
		local dloss_doutput = criterion:backward(batch_outputs,target)
		network:backward(input, dloss_doutput)

		-- update confusion matrix
		train_confusion:batchAdd(batch_outputs,target)

		return batch_loss, gradParameters
	end

	-----------------------training 
	local train_losses = {} -- training losses for each epoch
	local test_losses = {} -- testing losses
	local test_errs = {} -- testing error
	for epoch = 1, maxEpochs do
		-- batch processing parameters
    	batch_counter = 1
    	print(string.format('Epoch No:%d. (Max=%d)',epoch,maxEpochs))

    	-- maxIterations = 10
		for iteration= 1,maxIterations do
			xlua.progress(iteration,maxIterations) -- progress bar

			local _,minibatch_loss = optimMethod(feval,parameters, optimState)

			-- print loss at certain time; don't have to print all the time
			if iteration%10 ==0 then
				print('Training loss:'..minibatch_loss[1])
			end

			-- update losses for each epoch
			if iteration == maxIterations then
				train_losses[#train_losses + 1] = minibatch_loss[1]

				-- print
				train_confusion:updateValids()
				print('Training confusion matrix')
				print(train_confusion)
				print('Training accuracy:' .. train_confusion.totalValid * 100)
			end
		end
		-- reset confusion matrix
		train_confusion:zero()

		-- testing
		test_err, test_loss = test_predictor(network,criterion, testing_dataset, classes, classes_names)
		test_errs[#test_errs +1] = test_err
		test_losses[#test_losses + 1] = test_loss

		-- print		        
        print( "----------------------" )
        print( "Index Label Prediction" )
        print("Test loss:" .. test_loss)
       	print("Test error:" .. test_err)
       	print("Test accuracy:" .. (1-test_err)*100)
        print( "----------------------" )

		-- collect the garbage in case; update on 10/28/15 no need to do collect garbage anymore according to the official site
		-- collectgarbage()
	end

	---------------------------------- once this training is done, plot 
	-- the losses
	gnuplot.plot({
	  'train-loss',
	  torch.range(1, #train_losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
	  torch.Tensor(train_losses),           -- y-coordinates (the training losses)
	  '-'},
	  {
	  'test-loss',
	  torch.range(1, #test_losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
	  torch.Tensor(test_losses),           -- y-coordinates (the training losses)
	  '-'}
	 )
  	gnuplot.title('Training error')
	gnuplot.xlabel('Number of epochs')
	gnuplot.ylabel('Loss')

	-- the error
	gnuplot.figure()
	gnuplot.plot({
	  'test-error',
	  torch.range(1, #test_errs),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
	  torch.Tensor(test_errs),           -- y-coordinates (the training losses)
	  '-'}
	 )
  	gnuplot.title('Classification Error')
	gnuplot.xlabel('Number of epochs')
	gnuplot.ylabel('Error')
end

--test the network
function test_predictor(predictor, criterion, test_dataset, classes, classes_names)

        local mistakes = 0
        local tested_samples = 0
        local test_loss = 0 -- calculate the loss for testing example

        -- loop
        for i=1,test_dataset:size() do

               local input  = test_dataset[i][1]
               local class_id = test_dataset[i][2]

               --cuda
               input = input:float():cuda()
               local label = torch.Tensor(1):fill(class_id):float():cuda()
        		
        	   -- calculate the probability	
               local responses_per_class  =  predictor:forward(input) 
               local probabilites_per_class = torch.exp(responses_per_class)
               local probability, prediction = torch.max(probabilites_per_class, 1) 

               -- update loss
               test_loss = test_loss + criterion:forward(predictor:forward(input),label)

                      
               -- finding mismatch
               if prediction[1] ~= class_id then
                      mistakes = mistakes + 1
                      -- local label = classes_names[ classes[class_id] ]
                      -- local predicted_label = classes_names[ classes[prediction[1] ] ]
                      -- print(i , label , predicted_label )
               end

               tested_samples = tested_samples + 1
        end

        -- update
        local test_err =  mistakes/tested_samples
        test_loss = test_loss / tested_samples -- get the average loss
        return test_err,test_loss
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
	local network = simple_model(#classes)
	s_training_dataset = shuffleData(training_dataset)

    -- determine the optimization method
    local optimState, optimMethod = select_optim(params.optim)
    local parameters, gradParameters = network:getParameters() -- get the parameters of the network

	train_network(network,s_training_dataset, optimMethod, optimState, parameters, gradParameters,testing_dataset, classes, classes_names)
end

--run 
main()








































